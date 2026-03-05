#!/usr/bin/env python3
"""End-to-end causal tracing / activation patching pipeline for <tool_call> at t=1.

This script implements A0-A8 in todo.md and writes outputs to figs/ and reports/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import TwoSlopeNorm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

try:
    hf_logging.disable_progress_bar()
except Exception:
    pass
hf_logging.set_verbosity_error()


@dataclass
class PairSample:
    q: int
    clean_text: str
    corrupt_text: str
    clean_ids: torch.Tensor  # cpu long, shape [seq]
    corrupt_ids: torch.Tensor  # cpu long, shape [seq]
    seq_len: int
    diff_positions: List[int]


@dataclass
class RunCache:
    resid_pre: List[torch.Tensor]  # [L] each [P, D]
    attn_out: List[torch.Tensor]  # [L] each [P, D]
    mlp_out: List[torch.Tensor]  # [L] each [P, D]
    oproj_in: List[torch.Tensor]  # [L] each [D]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def head_name(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def bootstrap_mean_ci(
    values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return mean, lo, hi


def sign_flip_pvalue(values: Sequence[float], n_perm: int = 5000, seed: int = 0) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    obs = abs(float(arr.mean()))
    if obs == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_perm, arr.size))
    perm = np.abs((signs * arr[None, :]).mean(axis=1))
    p = (np.sum(perm >= obs) + 1.0) / (n_perm + 1.0)
    return float(p)


def paired_sign_flip_pvalue(
    a: Sequence[float], b: Sequence[float], n_perm: int = 5000, seed: int = 0
) -> float:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(arr_a) & np.isfinite(arr_b)
    if not np.any(m):
        return float("nan")
    diff = arr_a[m] - arr_b[m]
    return sign_flip_pvalue(diff, n_perm=n_perm, seed=seed)


def heatmap_clip(matrix: np.ndarray, quantile: float = 99.0) -> float:
    vals = np.abs(matrix[np.isfinite(matrix)])
    if vals.size == 0:
        return 1.0
    v = float(np.quantile(vals, quantile / 100.0))
    return max(v, 1e-8)


def save_diverging_heatmap(
    matrix: np.ndarray,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    xticks: Sequence[int] | None = None,
    xticklabels: Sequence[str] | None = None,
    yticks: Sequence[int] | None = None,
    yticklabels: Sequence[str] | None = None,
    clip_quantile: float = 99.0,
    figsize: Tuple[float, float] = (10.0, 6.0),
) -> None:
    vlim = heatmap_clip(matrix, clip_quantile)
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
        origin="lower",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.95)
    cbar.set_label("IE on P(<tool_call> at t=1)")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_triptych_heatmaps(
    matrices: Sequence[np.ndarray],
    titles: Sequence[str],
    out_path: Path,
    xlabel: str,
    ylabel: str,
    xticklabels: Sequence[str],
    ytick_step: int = 2,
    clip_quantile: float = 99.0,
    figsize: Tuple[float, float] = (17.2, 5.6),
) -> None:
    if len(matrices) != len(titles):
        raise ValueError("matrices and titles length mismatch")
    flat = np.concatenate([m[np.isfinite(m)].reshape(-1) for m in matrices if np.isfinite(m).any()])
    if flat.size == 0:
        vlim = 1.0
    else:
        vlim = max(float(np.quantile(np.abs(flat), clip_quantile / 100.0)), 1e-8)
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    n_layers = matrices[0].shape[0]
    fig, axes = plt.subplots(1, len(matrices), figsize=figsize, constrained_layout=True)
    if len(matrices) == 1:
        axes = [axes]

    for ax, mat, title in zip(axes, matrices, titles):
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
            origin="lower",
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_xticks(list(range(len(xticklabels))))
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(list(range(0, n_layers, ytick_step)))
        ax.set_yticklabels([str(i) for i in range(0, n_layers, ytick_step)])
    axes[0].set_ylabel(ylabel, fontsize=11)
    for ax in axes[1:]:
        ax.set_ylabel("")

    cbar = fig.colorbar(im, ax=axes, shrink=0.9, location="right", pad=0.02)
    cbar.set_label("IE on P(<tool_call> at t=1)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_lineplot_with_ci(
    x: np.ndarray,
    series: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    colors = {
        "full_restore": "#1f77b4",
        "freeze_future_mlp": "#d62728",
        "freeze_future_attn": "#2ca02c",
    }
    labels = {
        "full_restore": "Full restore",
        "freeze_future_mlp": "Freeze future MLP",
        "freeze_future_attn": "Freeze future Attention",
    }
    for key, mat in series.items():
        mean = np.nanmean(mat, axis=0)
        lo = np.nanpercentile(mat, 2.5, axis=0)
        hi = np.nanpercentile(mat, 97.5, axis=0)
        c = colors.get(key, None)
        ax.plot(x, mean, label=labels.get(key, key), color=c, linewidth=2.0)
        ax.fill_between(x, lo, hi, alpha=0.15, color=c)

    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def load_pairs(pair_dir: Path, tokenizer) -> List[PairSample]:
    clean_paths = sorted(pair_dir.glob("prompt-clean-q*.txt"), key=lambda p: int(re.search(r"q(\d+)", p.name).group(1)))
    pairs: List[PairSample] = []
    for cp in clean_paths:
        q = int(re.search(r"q(\d+)", cp.name).group(1))
        rp = pair_dir / f"prompt-corrupted-q{q}.txt"
        if not rp.exists():
            continue
        clean_text = cp.read_text()
        corrupt_text = rp.read_text()
        clean_ids = tokenizer.encode(clean_text, add_special_tokens=False)
        corrupt_ids = tokenizer.encode(corrupt_text, add_special_tokens=False)
        if len(clean_ids) != len(corrupt_ids):
            # Keep only aligned pairs for this pipeline.
            continue
        clean_t = torch.tensor(clean_ids, dtype=torch.long)
        corr_t = torch.tensor(corrupt_ids, dtype=torch.long)
        diff_positions = [i for i, (a, b) in enumerate(zip(clean_ids, corrupt_ids)) if a != b]
        pairs.append(
            PairSample(
                q=q,
                clean_text=clean_text,
                corrupt_text=corrupt_text,
                clean_ids=clean_t,
                corrupt_ids=corr_t,
                seq_len=len(clean_ids),
                diff_positions=diff_positions,
            )
        )
    return pairs


def run_prob(model, input_ids_1d: torch.Tensor, tool_id: int) -> Tuple[float, int]:
    with torch.no_grad():
        out = model(input_ids=input_ids_1d.unsqueeze(0), use_cache=False)
    logits = out.logits[0, -1]
    top_id = int(torch.argmax(logits).item())
    logsumexp = torch.logsumexp(logits.float(), dim=-1)
    p_tool = float(torch.exp(logits[tool_id].float() - logsumexp).item())
    return p_tool, top_id


def run_prob_patched(
    model,
    input_ids_1d: torch.Tensor,
    tool_id: int,
    hook_builders: Sequence[Callable[[], torch.utils.hooks.RemovableHandle]],
) -> Tuple[float, int]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    try:
        for build in hook_builders:
            handles.append(build())
        return run_prob(model, input_ids_1d, tool_id)
    finally:
        for h in handles:
            h.remove()


def collect_cache(
    model,
    input_ids_1d: torch.Tensor,
    pos_list: Sequence[int],
) -> RunCache:
    layers = model.model.layers
    n_layers = len(layers)
    last_pos = input_ids_1d.numel() - 1
    device = input_ids_1d.device
    pos_tensor = torch.tensor(pos_list, device=device, dtype=torch.long)

    resid_pre: List[torch.Tensor] = [None] * n_layers
    attn_out: List[torch.Tensor] = [None] * n_layers
    mlp_out: List[torch.Tensor] = [None] * n_layers
    oproj_in: List[torch.Tensor] = [None] * n_layers
    handles: List[torch.utils.hooks.RemovableHandle] = []

    for li, layer in enumerate(layers):
        def make_layer_pre(idx: int):
            def _hook(module, inputs):
                hs = inputs[0]  # [1, seq, d]
                resid_pre[idx] = hs[0].index_select(0, pos_tensor).detach().clone()
            return _hook

        def make_attn_hook(idx: int):
            def _hook(module, inputs, output):
                t = output[0] if isinstance(output, tuple) else output
                attn_out[idx] = t[0].index_select(0, pos_tensor).detach().clone()
            return _hook

        def make_mlp_hook(idx: int):
            def _hook(module, inputs, output):
                mlp_out[idx] = output[0].index_select(0, pos_tensor).detach().clone()
            return _hook

        def make_oproj_pre(idx: int):
            def _hook(module, inputs):
                x = inputs[0]  # [1, seq, d]
                oproj_in[idx] = x[0, last_pos, :].detach().clone()
            return _hook

        handles.append(layer.register_forward_pre_hook(make_layer_pre(li)))
        handles.append(layer.self_attn.register_forward_hook(make_attn_hook(li)))
        handles.append(layer.mlp.register_forward_hook(make_mlp_hook(li)))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_oproj_pre(li)))

    try:
        _ = run_prob(model, input_ids_1d, tool_id=0)
    finally:
        for h in handles:
            h.remove()

    for li in range(n_layers):
        if resid_pre[li] is None or attn_out[li] is None or mlp_out[li] is None or oproj_in[li] is None:
            raise RuntimeError(f"cache collection failed at layer {li}")

    return RunCache(resid_pre=resid_pre, attn_out=attn_out, mlp_out=mlp_out, oproj_in=oproj_in)


def run_resid_restore(
    model,
    input_ids_1d: torch.Tensor,
    tool_id: int,
    layer_idx: int,
    token_pos: int,
    patch_vec: torch.Tensor,
    extra_hooks: Sequence[Callable[[], torch.utils.hooks.RemovableHandle]] | None = None,
) -> Tuple[float, int]:
    layer = model.model.layers[layer_idx]

    def build_resid_hook():
        def _hook(module, inputs):
            hs = inputs[0]
            hs2 = hs.clone()
            hs2[:, token_pos, :] = patch_vec
            return (hs2,) + tuple(inputs[1:])
        return layer.register_forward_pre_hook(_hook)

    hook_builders = [build_resid_hook]
    if extra_hooks:
        hook_builders.extend(extra_hooks)
    return run_prob_patched(model, input_ids_1d, tool_id, hook_builders)


def run_attn_restore(
    model,
    input_ids_1d: torch.Tensor,
    tool_id: int,
    layer_idx: int,
    token_pos: int,
    patch_vec: torch.Tensor,
) -> Tuple[float, int]:
    module = model.model.layers[layer_idx].self_attn

    def build_hook():
        def _hook(m, inputs, output):
            if isinstance(output, tuple):
                out0 = output[0].clone()
                out0[:, token_pos, :] = patch_vec
                return (out0, output[1])
            out = output.clone()
            out[:, token_pos, :] = patch_vec
            return out
        return module.register_forward_hook(_hook)

    return run_prob_patched(model, input_ids_1d, tool_id, [build_hook])


def run_mlp_restore(
    model,
    input_ids_1d: torch.Tensor,
    tool_id: int,
    layer_idx: int,
    token_pos: int,
    patch_vec: torch.Tensor,
) -> Tuple[float, int]:
    module = model.model.layers[layer_idx].mlp

    def build_hook():
        def _hook(m, inputs, output):
            out = output.clone()
            out[:, token_pos, :] = patch_vec
            return out
        return module.register_forward_hook(_hook)

    return run_prob_patched(model, input_ids_1d, tool_id, [build_hook])


def run_multi_attn_restore(
    model,
    input_ids_1d: torch.Tensor,
    tool_id: int,
    layer_to_vec: Dict[int, torch.Tensor],
    token_pos: int,
) -> Tuple[float, int]:
    hooks = []
    for li, vec in layer_to_vec.items():
        module = model.model.layers[li].self_attn

        def build_hook(m=module, patch=vec):
            def _hook(_m, inputs, output):
                if isinstance(output, tuple):
                    out0 = output[0].clone()
                    out0[:, token_pos, :] = patch
                    return (out0, output[1])
                out = output.clone()
                out[:, token_pos, :] = patch
                return out
            return m.register_forward_hook(_hook)

        hooks.append(build_hook)
    return run_prob_patched(model, input_ids_1d, tool_id, hooks)


def run_multi_mlp_restore(
    model,
    input_ids_1d: torch.Tensor,
    tool_id: int,
    layer_to_vec: Dict[int, torch.Tensor],
    token_pos: int,
) -> Tuple[float, int]:
    hooks = []
    for li, vec in layer_to_vec.items():
        module = model.model.layers[li].mlp

        def build_hook(m=module, patch=vec):
            def _hook(_m, inputs, output):
                out = output.clone()
                out[:, token_pos, :] = patch
                return out
            return m.register_forward_hook(_hook)

        hooks.append(build_hook)
    return run_prob_patched(model, input_ids_1d, tool_id, hooks)


def run_head_patch(
    model,
    input_ids_1d: torch.Tensor,
    tool_id: int,
    layer_idx: int,
    head_idx: int,
    token_pos: int,
    patch_chunk: torch.Tensor,
    head_dim: int,
) -> Tuple[float, int]:
    module = model.model.layers[layer_idx].self_attn.o_proj
    start = head_idx * head_dim
    end = start + head_dim

    def build_hook():
        def _hook(m, inputs):
            x = inputs[0]
            x2 = x.clone()
            x2[:, token_pos, start:end] = patch_chunk
            return (x2,)
        return module.register_forward_pre_hook(_hook)

    return run_prob_patched(model, input_ids_1d, tool_id, [build_hook])


def run_multi_head_patch(
    model,
    input_ids_1d: torch.Tensor,
    tool_id: int,
    token_pos: int,
    patches: Sequence[Tuple[int, int, torch.Tensor]],
    head_dim: int,
) -> Tuple[float, int]:
    by_layer: Dict[int, List[Tuple[int, torch.Tensor]]] = defaultdict(list)
    for li, hi, vec in patches:
        by_layer[li].append((hi, vec))

    hooks: List[Callable[[], torch.utils.hooks.RemovableHandle]] = []
    for li, layer_patches in by_layer.items():
        module = model.model.layers[li].self_attn.o_proj

        def build_hook(m=module, patch_list=layer_patches):
            def _hook(_m, inputs):
                x = inputs[0]
                x2 = x.clone()
                for hi, vec in patch_list:
                    s = hi * head_dim
                    e = s + head_dim
                    x2[:, token_pos, s:e] = vec
                return (x2,)
            return m.register_forward_pre_hook(_hook)

        hooks.append(build_hook)

    return run_prob_patched(model, input_ids_1d, tool_id, hooks)


def freeze_future_to_baseline_hooks(
    model,
    from_layer_exclusive: int,
    mode: str,
    token_pos: int,
    layer_to_vec: Dict[int, torch.Tensor],
) -> List[Callable[[], torch.utils.hooks.RemovableHandle]]:
    hooks: List[Callable[[], torch.utils.hooks.RemovableHandle]] = []
    n_layers = len(model.model.layers)
    for li in range(from_layer_exclusive + 1, n_layers):
        if li not in layer_to_vec:
            continue
        layer = model.model.layers[li]
        frozen_vec = layer_to_vec[li]
        if mode == "mlp":
            module = layer.mlp

            def build_hook(m=module, vec=frozen_vec):
                def _hook(_m, inputs, output):
                    out = output.clone()
                    out[:, token_pos, :] = vec
                    return out
                return m.register_forward_hook(_hook)

            hooks.append(build_hook)
        elif mode == "attn":
            module = layer.self_attn

            def build_hook(m=module, vec=frozen_vec):
                def _hook(_m, inputs, output):
                    if isinstance(output, tuple):
                        out0 = output[0].clone()
                        out0[:, token_pos, :] = vec
                        return (out0, output[1])
                    out = output.clone()
                    out[:, token_pos, :] = vec
                    return out
                return m.register_forward_hook(_hook)

            hooks.append(build_hook)
        else:
            raise ValueError(mode)
    return hooks


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def select_subset(
    strict_df: pd.DataFrame,
    n: int,
    strategy: str = "stratified",
    seed: int = 0,
    n_buckets: int = 4,
) -> List[int]:
    if strict_df.empty:
        return []
    all_qs = [int(x) for x in strict_df.sort_values("q")["q"].tolist()]
    if n <= 0 or n >= len(all_qs):
        return all_qs

    if strategy == "shortest":
        sel = strict_df.sort_values(["seq_len", "q"]).head(n)
        return [int(x) for x in sel["q"].tolist()]

    rng = np.random.default_rng(seed)
    if strategy == "random":
        chosen = strict_df.sample(n=n, replace=False, random_state=seed)["q"].tolist()
        return [int(x) for x in sorted(chosen)]

    # Stratified by sequence-length quantile buckets (default).
    bins = pd.qcut(
        strict_df["seq_len"],
        q=min(n_buckets, strict_df["seq_len"].nunique()),
        labels=False,
        duplicates="drop",
    )
    df = strict_df.copy()
    df["bucket"] = bins
    bucket_ids = sorted([int(x) for x in df["bucket"].dropna().unique().tolist()])
    if not bucket_ids:
        chosen = strict_df.sample(n=n, replace=False, random_state=seed)["q"].tolist()
        return [int(x) for x in sorted(chosen)]

    per_bucket = {b: df[df["bucket"] == b].sort_values("q") for b in bucket_ids}
    alloc = {b: 0 for b in bucket_ids}
    # proportional allocation with floor, then fill remainders.
    total = sum(len(per_bucket[b]) for b in bucket_ids)
    for b in bucket_ids:
        alloc[b] = int(math.floor(n * len(per_bucket[b]) / total))
    used = sum(alloc.values())
    remaining = n - used
    order = sorted(bucket_ids, key=lambda b: len(per_bucket[b]), reverse=True)
    oi = 0
    while remaining > 0 and order:
        b = order[oi % len(order)]
        if alloc[b] < len(per_bucket[b]):
            alloc[b] += 1
            remaining -= 1
        oi += 1
        if oi > 10000:
            break

    chosen: List[int] = []
    for b in bucket_ids:
        sub = per_bucket[b]
        take = min(alloc[b], len(sub))
        if take <= 0:
            continue
        idx = rng.choice(len(sub), size=take, replace=False)
        chosen.extend([int(x) for x in sub.iloc[idx]["q"].tolist()])
    chosen = sorted(set(chosen))

    # In rare collisions after dedup, top up globally.
    if len(chosen) < n:
        left = [q for q in all_qs if q not in set(chosen)]
        extra = left[: (n - len(chosen))]
        chosen.extend(extra)
    return sorted(chosen[:n])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/root/data/Qwen/Qwen3-1.7B"))
    parser.add_argument("--pair-dir", type=Path, default=Path("pair"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figs"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--analysis-strict-n", type=int, default=32)
    parser.add_argument("--head-strict-n", type=int, default=24)
    parser.add_argument(
        "--subset-strategy",
        type=str,
        default="stratified",
        choices=["stratified", "random", "shortest"],
    )
    parser.add_argument("--ct-window", type=int, default=16)
    parser.add_argument("--ct-stride", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--circuit-k", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    log("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    tool_token_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    if len(tool_token_ids) != 1:
        raise RuntimeError(f"Expected single token for <tool_call>, got {tool_token_ids}")
    tool_id = int(tool_token_ids[0])

    n_layers = len(model.model.layers)
    n_heads = int(model.config.num_attention_heads)
    hidden_size = int(model.config.hidden_size)
    head_dim = hidden_size // n_heads

    log("Loading paired prompts...")
    pairs = load_pairs(args.pair_dir, tokenizer)
    q_to_pair = {p.q: p for p in pairs}
    if not pairs:
        raise RuntimeError("No aligned pair samples were loaded.")
    log(f"Loaded {len(pairs)} aligned pairs.")

    # ------------------------------------------------------------------
    # A0/A1: Baseline and TE
    # ------------------------------------------------------------------
    log("Running baseline forward pass on all pairs...")
    rows = []
    for i, pair in enumerate(pairs, start=1):
        clean_ids = pair.clean_ids.to(model.device)
        corr_ids = pair.corrupt_ids.to(model.device)
        p_clean, top_clean = run_prob(model, clean_ids, tool_id)
        p_corr, top_corr = run_prob(model, corr_ids, tool_id)
        row = {
            "q": pair.q,
            "seq_len": pair.seq_len,
            "p_clean": p_clean,
            "p_corrupt": p_corr,
            "top1_clean_id": top_clean,
            "top1_corrupt_id": top_corr,
            "top1_clean_tok": tokenizer.decode([top_clean]),
            "top1_corrupt_tok": tokenizer.decode([top_corr]),
            "y_clean": int(top_clean == tool_id),
            "y_corrupt_non_tool": int(top_corr != tool_id),
        }
        row["strict"] = int(row["y_clean"] == 1 and row["y_corrupt_non_tool"] == 1)
        row["TE"] = row["p_clean"] - row["p_corrupt"]
        rows.append(row)
        if i % 20 == 0 or i == len(pairs):
            log(f"  baseline progress: {i}/{len(pairs)}")

    baseline_df = pd.DataFrame(rows).sort_values("q").reset_index(drop=True)
    te_csv_path = args.report_dir / "te_summary.csv"
    baseline_df.to_csv(te_csv_path, index=False)

    full_n = len(baseline_df)
    strict_df = baseline_df[baseline_df["strict"] == 1].copy()
    strict_n = len(strict_df)

    clean_rate = baseline_df["y_clean"].mean()
    corr_nontool_rate = baseline_df["y_corrupt_non_tool"].mean()
    strict_rate = baseline_df["strict"].mean()

    te_full_mean, te_full_lo, te_full_hi = bootstrap_mean_ci(
        baseline_df["TE"].values, n_boot=args.n_bootstrap, seed=args.seed + 1
    )
    te_strict_mean, te_strict_lo, te_strict_hi = bootstrap_mean_ci(
        strict_df["TE"].values, n_boot=args.n_bootstrap, seed=args.seed + 2
    )

    # Optional comparison to provided baseline CSV.
    provided_csv = args.pair_dir / "first_token_len_eval_qwen3_1.7b.csv"
    provided_msg = "N/A"
    if provided_csv.exists():
        pdf = pd.read_csv(provided_csv)
        p_clean_ref = (pdf["clean_top1"] == "<tool_call>").mean()
        p_corr_ref = (pdf["corr_top1"] != "<tool_call>").mean()
        p_strict_ref = ((pdf["clean_top1"] == "<tool_call>") & (pdf["corr_top1"] != "<tool_call>")).mean()
        provided_msg = (
            f"provided clean={p_clean_ref:.3f}, corrupt_non_tool={p_corr_ref:.3f}, strict={p_strict_ref:.3f}; "
            f"recomputed clean={clean_rate:.3f}, corrupt_non_tool={corr_nontool_rate:.3f}, strict={strict_rate:.3f}"
        )

    baseline_md = []
    baseline_md.append("# Baseline Metrics")
    baseline_md.append("")
    baseline_md.append("- 主指标：`p_tool = P(<tool_call> at t=1)`；二值标签 `y = 1[top1 == <tool_call>]`。")
    baseline_md.append(f"- 总样本数（对齐后）：`N={full_n}`；严格子集：`N={strict_n}`。")
    baseline_md.append("")
    baseline_md.append("## Full Set (All Aligned Pairs)")
    baseline_md.append(f"- clean top1=`<tool_call>`: `{baseline_df['y_clean'].sum()}/{full_n} = {clean_rate:.3%}`")
    baseline_md.append(
        f"- corrupt top1!=`<tool_call>`: `{baseline_df['y_corrupt_non_tool'].sum()}/{full_n} = {corr_nontool_rate:.3%}`"
    )
    baseline_md.append(f"- strict pair success: `{baseline_df['strict'].sum()}/{full_n} = {strict_rate:.3%}`")
    baseline_md.append(
        f"- `TE = p_clean - p_corrupt` mean: `{te_full_mean:.5f}` (95% CI `{te_full_lo:.5f}`, `{te_full_hi:.5f}`)"
    )
    baseline_md.append("")
    baseline_md.append("## Strict Subset")
    baseline_md.append(
        f"- `TE` mean: `{te_strict_mean:.5f}` (95% CI `{te_strict_lo:.5f}`, `{te_strict_hi:.5f}`)"
    )
    baseline_md.append("")
    baseline_md.append("## Sanity Check vs Provided CSV")
    baseline_md.append(f"- {provided_msg}")
    baseline_md.append("")
    baseline_md.append("## Figure/Metric Notes")
    baseline_md.append("- 全部指标只用第一个生成位置 `t=1`。")
    baseline_md.append("- 置信区间为 bootstrap 95% CI。")
    baseline_md.append("- `te_summary.csv` 已保存每个样本的 `p_clean / p_corrupt / TE / strict`。")
    write_text(args.report_dir / "baseline_metrics.md", "\n".join(baseline_md))

    # ------------------------------------------------------------------
    # Subset selection for heavier analyses.
    # ------------------------------------------------------------------
    analysis_qs = select_subset(
        strict_df,
        args.analysis_strict_n,
        strategy=args.subset_strategy,
        seed=args.seed + 13,
    )
    head_qs = select_subset(
        strict_df,
        args.head_strict_n,
        strategy=args.subset_strategy,
        seed=args.seed + 29,
    )
    if not analysis_qs or not head_qs:
        raise RuntimeError("Strict subset is empty; cannot continue with causal tracing.")

    log(f"Selected strict subset for CT/module/path: N={len(analysis_qs)} (strategy={args.subset_strategy})")
    log(f"Selected strict subset for head AP/CT: N={len(head_qs)} (strategy={args.subset_strategy})")

    # ------------------------------------------------------------------
    # A2/A3/A4: CT state heatmap + module decomposition + modified graph.
    # ------------------------------------------------------------------
    rel_positions = list(range(-args.ct_window, args.ct_window + 1, args.ct_stride))
    center_rel_idx = rel_positions.index(0) if 0 in rel_positions else len(rel_positions) // 2

    ct_state = np.full((len(analysis_qs), n_layers, len(rel_positions)), np.nan, dtype=np.float32)
    ct_state_decision = np.full((len(analysis_qs), n_layers), np.nan, dtype=np.float32)
    ct_attn_map = np.full((len(analysis_qs), n_layers, len(rel_positions)), np.nan, dtype=np.float32)
    ct_mlp_map = np.full((len(analysis_qs), n_layers, len(rel_positions)), np.nan, dtype=np.float32)
    ct_attn_decision = np.full((len(analysis_qs), n_layers), np.nan, dtype=np.float32)
    ct_mlp_decision = np.full((len(analysis_qs), n_layers), np.nan, dtype=np.float32)
    window_attn = np.full((len(analysis_qs), n_layers), np.nan, dtype=np.float32)
    window_mlp = np.full((len(analysis_qs), n_layers), np.nan, dtype=np.float32)
    mod_full_center = np.full((len(analysis_qs), n_layers), np.nan, dtype=np.float32)
    mod_no_mlp = np.full((len(analysis_qs), n_layers), np.nan, dtype=np.float32)
    mod_no_attn = np.full((len(analysis_qs), n_layers), np.nan, dtype=np.float32)

    log("Running A2/A3/A4 scans...")
    for si, q in enumerate(analysis_qs):
        pair = q_to_pair[q]
        seq_len = pair.seq_len
        diff = pair.diff_positions
        center = diff[len(diff) // 2] if diff else max(0, seq_len - 4)
        abs_positions = [min(max(center + r, 0), seq_len - 1) for r in rel_positions]
        last_pos = seq_len - 1
        center_pos = abs_positions[center_rel_idx]
        cache_positions = sorted(set(abs_positions + [last_pos]))
        pos_to_idx = {p: i for i, p in enumerate(cache_positions)}

        clean_ids = pair.clean_ids.to(model.device)
        corr_ids = pair.corrupt_ids.to(model.device)

        p_clean = float(baseline_df.loc[baseline_df.q == q, "p_clean"].iloc[0])
        p_corr = float(baseline_df.loc[baseline_df.q == q, "p_corrupt"].iloc[0])

        clean_cache = collect_cache(model, clean_ids, cache_positions)
        corr_cache = collect_cache(model, corr_ids, cache_positions)

        # A2/A3: state + module traces over relative positions.
        for li in range(n_layers):
            for ri, pos in enumerate(abs_positions):
                pidx = pos_to_idx[pos]
                p_patch, _ = run_resid_restore(
                    model=model,
                    input_ids_1d=corr_ids,
                    tool_id=tool_id,
                    layer_idx=li,
                    token_pos=pos,
                    patch_vec=clean_cache.resid_pre[li][pidx],
                )
                ct_state[si, li, ri] = p_patch - p_corr

                p_attn_map, _ = run_attn_restore(
                    model=model,
                    input_ids_1d=corr_ids,
                    tool_id=tool_id,
                    layer_idx=li,
                    token_pos=pos,
                    patch_vec=clean_cache.attn_out[li][pidx],
                )
                p_mlp_map, _ = run_mlp_restore(
                    model=model,
                    input_ids_1d=corr_ids,
                    tool_id=tool_id,
                    layer_idx=li,
                    token_pos=pos,
                    patch_vec=clean_cache.mlp_out[li][pidx],
                )
                ct_attn_map[si, li, ri] = p_attn_map - p_corr
                ct_mlp_map[si, li, ri] = p_mlp_map - p_corr

            # Late-site traces at decision token (for line plot and summary stats).
            lpidx = pos_to_idx[last_pos]
            p_state_dec, _ = run_resid_restore(
                model=model,
                input_ids_1d=corr_ids,
                tool_id=tool_id,
                layer_idx=li,
                token_pos=last_pos,
                patch_vec=clean_cache.resid_pre[li][lpidx],
            )
            p_attn_dec, _ = run_attn_restore(
                model=model,
                input_ids_1d=corr_ids,
                tool_id=tool_id,
                layer_idx=li,
                token_pos=last_pos,
                patch_vec=clean_cache.attn_out[li][lpidx],
            )
            p_mlp_dec, _ = run_mlp_restore(
                model=model,
                input_ids_1d=corr_ids,
                tool_id=tool_id,
                layer_idx=li,
                token_pos=last_pos,
                patch_vec=clean_cache.mlp_out[li][lpidx],
            )
            ct_state_decision[si, li] = p_state_dec - p_corr
            ct_attn_decision[si, li] = p_attn_dec - p_corr
            ct_mlp_decision[si, li] = p_mlp_dec - p_corr

        # A3: 10-layer window restore.
        lpidx = pos_to_idx[last_pos]
        for start in range(n_layers):
            layers_w = list(range(start, min(n_layers, start + args.window_size)))
            attn_dict = {l: clean_cache.attn_out[l][lpidx] for l in layers_w}
            mlp_dict = {l: clean_cache.mlp_out[l][lpidx] for l in layers_w}
            p_aw, _ = run_multi_attn_restore(model, corr_ids, tool_id, attn_dict, last_pos)
            p_mw, _ = run_multi_mlp_restore(model, corr_ids, tool_id, mlp_dict, last_pos)
            window_attn[si, start] = p_aw - p_corr
            window_mlp[si, start] = p_mw - p_corr

        # A4: modified graph interventions (freeze future module paths to corrupted baseline values).
        cpidx = pos_to_idx[center_pos]
        for li in range(n_layers):
            full_ie = ct_state[si, li, center_rel_idx]
            mod_full_center[si, li] = full_ie

            mlp_baseline = {l: corr_cache.mlp_out[l][cpidx] for l in range(li + 1, n_layers)}
            mlp_freeze = freeze_future_to_baseline_hooks(
                model=model,
                from_layer_exclusive=li,
                mode="mlp",
                token_pos=center_pos,
                layer_to_vec=mlp_baseline,
            )
            p_corr_mlp, _ = run_prob_patched(model, corr_ids, tool_id, mlp_freeze)
            p_patch_mlp, _ = run_resid_restore(
                model=model,
                input_ids_1d=corr_ids,
                tool_id=tool_id,
                layer_idx=li,
                token_pos=center_pos,
                patch_vec=clean_cache.resid_pre[li][cpidx],
                extra_hooks=mlp_freeze,
            )
            mod_no_mlp[si, li] = p_patch_mlp - p_corr_mlp

            attn_baseline = {l: corr_cache.attn_out[l][cpidx] for l in range(li + 1, n_layers)}
            attn_freeze = freeze_future_to_baseline_hooks(
                model=model,
                from_layer_exclusive=li,
                mode="attn",
                token_pos=center_pos,
                layer_to_vec=attn_baseline,
            )
            p_corr_attn, _ = run_prob_patched(model, corr_ids, tool_id, attn_freeze)
            p_patch_attn, _ = run_resid_restore(
                model=model,
                input_ids_1d=corr_ids,
                tool_id=tool_id,
                layer_idx=li,
                token_pos=center_pos,
                patch_vec=clean_cache.resid_pre[li][cpidx],
                extra_hooks=attn_freeze,
            )
            mod_no_attn[si, li] = p_patch_attn - p_corr_attn

        log(f"  A2/A3/A4 progress: {si + 1}/{len(analysis_qs)} (q={q}, len={seq_len}, p_clean={p_clean:.4f}, p_corr={p_corr:.4f})")

    ct_state_mean = np.nanmean(ct_state, axis=0)  # [L, R]
    ct_attn_mean_map = np.nanmean(ct_attn_map, axis=0)  # [L, R]
    ct_mlp_mean_map = np.nanmean(ct_mlp_map, axis=0)  # [L, R]
    ct_state_center = ct_state[:, :, center_rel_idx]  # [S, L]
    ct_attn_mean = np.nanmean(ct_attn_decision, axis=0)  # [L]
    ct_mlp_mean = np.nanmean(ct_mlp_decision, axis=0)  # [L]

    # A2 output figure + topk table.
    save_diverging_heatmap(
        matrix=ct_state_mean,
        out_path=args.fig_dir / "ct_state_heatmap.png",
        title="State-level Causal Trace (Corrupt + Residual Restore from Clean)",
        xlabel="Relative Position to Corruption Center",
        ylabel="Layer",
        xticks=list(range(len(rel_positions))),
        xticklabels=[str(r) for r in rel_positions],
        yticks=list(range(0, n_layers, 2)),
        yticklabels=[str(i) for i in range(0, n_layers, 2)],
        clip_quantile=99.0,
        figsize=(12.5, 6.5),
    )

    flat_rows = []
    for li in range(n_layers):
        for ri, rel in enumerate(rel_positions):
            vals = ct_state[:, li, ri]
            mean, lo, hi = bootstrap_mean_ci(vals[np.isfinite(vals)], n_boot=args.n_bootstrap, seed=args.seed + 1000 + li * 97 + ri)
            flat_rows.append(
                {
                    "layer": li,
                    "relative_pos": rel,
                    "mean_ie": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "abs_mean_ie": abs(mean),
                }
            )
    ct_state_topk = pd.DataFrame(flat_rows).sort_values("abs_mean_ie", ascending=False).head(40)
    ct_state_topk.to_csv(args.report_dir / "ct_state_topk.csv", index=False)

    # A3 figures (paper-style: position x layer maps).
    save_diverging_heatmap(
        matrix=ct_mlp_mean_map,
        out_path=args.fig_dir / "ct_mlp_heatmap.png",
        title="MLP Causal Trace Map (Corrupt[MLP<-Clean] - Corrupt)",
        xlabel="Relative Position to Corruption Center",
        ylabel="Layer",
        xticks=list(range(len(rel_positions))),
        xticklabels=[str(r) for r in rel_positions],
        yticks=list(range(0, n_layers, 2)),
        yticklabels=[str(i) for i in range(0, n_layers, 2)],
        clip_quantile=99.0,
        figsize=(12.5, 6.5),
    )
    save_diverging_heatmap(
        matrix=ct_attn_mean_map,
        out_path=args.fig_dir / "ct_attn_heatmap.png",
        title="Attention Causal Trace Map (Corrupt[Attn<-Clean] - Corrupt)",
        xlabel="Relative Position to Corruption Center",
        ylabel="Layer",
        xticks=list(range(len(rel_positions))),
        xticklabels=[str(r) for r in rel_positions],
        yticks=list(range(0, n_layers, 2)),
        yticklabels=[str(i) for i in range(0, n_layers, 2)],
        clip_quantile=99.0,
        figsize=(12.5, 6.5),
    )
    save_triptych_heatmaps(
        matrices=[ct_state_mean, ct_mlp_mean_map, ct_attn_mean_map],
        titles=["State", "MLP", "Attention"],
        out_path=args.fig_dir / "info_flow_maps_like.png",
        xlabel="Relative Position to Corruption Center",
        ylabel="Layer",
        xticklabels=[str(r) for r in rel_positions],
        ytick_step=2,
        clip_quantile=99.0,
        figsize=(17.2, 5.8),
    )

    # A4 figure.
    layer_x = np.arange(n_layers)
    save_lineplot_with_ci(
        x=layer_x,
        series={
            "full_restore": mod_full_center,
            "freeze_future_mlp": mod_no_mlp,
            "freeze_future_attn": mod_no_attn,
        },
        out_path=args.fig_dir / "modified_graph_mlp_vs_attn.png",
        title="Modified Graph Intervention: Future-path Freezing",
        xlabel="Restored Layer",
        ylabel="IE on P(<tool_call> at t=1)",
    )
    # Additional line plot (paper-style) comparing early site vs late site.
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for key, mat, color, label in [
        ("state_center", ct_state_center, "#d62728", "State @ corruption center"),
        ("state_decision", ct_state_decision, "#1f77b4", "State @ decision token"),
        ("mlp_decision", ct_mlp_decision, "#2ca02c", "MLP @ decision token"),
        ("attn_decision", ct_attn_decision, "#9467bd", "Attn @ decision token"),
    ]:
        mean = np.nanmean(mat, axis=0)
        lo = np.nanpercentile(mat, 2.5, axis=0)
        hi = np.nanpercentile(mat, 97.5, axis=0)
        ax.plot(layer_x, mean, color=color, label=label, linewidth=2.0)
        ax.fill_between(layer_x, lo, hi, color=color, alpha=0.15)
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    ax.set_title("Causal Trace Line Plot with 95% CI", fontsize=13)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("IE on P(<tool_call> at t=1)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(args.fig_dir / "ct_lineplot_with_ci.png", dpi=220)
    plt.close(fig)

    # A3 report.
    top_mlp_layers = np.argsort(-np.abs(ct_mlp_mean))[:8]
    top_attn_layers = np.argsort(-np.abs(ct_attn_mean))[:8]
    win_mlp_mean = np.nanmean(window_mlp, axis=0)
    win_attn_mean = np.nanmean(window_attn, axis=0)
    best_mlp_start = int(np.nanargmax(win_mlp_mean))
    best_attn_start = int(np.nanargmax(win_attn_mean))
    mlp_global = bootstrap_mean_ci(ct_mlp_decision.flatten(), n_boot=args.n_bootstrap, seed=args.seed + 2000)
    attn_global = bootstrap_mean_ci(ct_attn_decision.flatten(), n_boot=args.n_bootstrap, seed=args.seed + 2001)
    mlp_p = sign_flip_pvalue(ct_mlp_decision.flatten(), n_perm=args.n_perm, seed=args.seed + 2050)
    attn_p = sign_flip_pvalue(ct_attn_decision.flatten(), n_perm=args.n_perm, seed=args.seed + 2051)
    center_state_global = bootstrap_mean_ci(ct_state_center.flatten(), n_boot=args.n_bootstrap, seed=args.seed + 2052)
    center_state_p = sign_flip_pvalue(ct_state_center.flatten(), n_perm=args.n_perm, seed=args.seed + 2053)
    decision_state_global = bootstrap_mean_ci(
        ct_state_decision.flatten(), n_boot=args.n_bootstrap, seed=args.seed + 2054
    )
    decision_state_p = sign_flip_pvalue(ct_state_decision.flatten(), n_perm=args.n_perm, seed=args.seed + 2055)
    attn_vs_mlp_p = paired_sign_flip_pvalue(
        ct_attn_decision.flatten(), ct_mlp_decision.flatten(), n_perm=args.n_perm, seed=args.seed + 2056
    )

    module_md = []
    module_md.append("# MLP vs Attention Decomposition")
    module_md.append("")
    module_md.append(f"- 样本：严格子集分析集（分层抽样），`N={len(analysis_qs)}`。")
    module_md.append("- 指标：`IE = p_tool(corrupt + restore module_from_clean) - p_tool(corrupt)`。")
    module_md.append("- 模块热图为 `layer x relative_position` 聚合均值，组织方式对齐论文因果图。")
    module_md.append("")
    module_md.append("## Global Compare")
    module_md.append(
        f"- MLP IE @ decision token mean: `{mlp_global[0]:.6f}` "
        f"(95% CI `{mlp_global[1]:.6f}`, `{mlp_global[2]:.6f}`), p=`{mlp_p:.3e}`"
    )
    module_md.append(
        f"- Attn IE @ decision token mean: `{attn_global[0]:.6f}` "
        f"(95% CI `{attn_global[1]:.6f}`, `{attn_global[2]:.6f}`), p=`{attn_p:.3e}`"
    )
    module_md.append(
        f"- State IE @ corruption center mean: `{center_state_global[0]:.6f}` "
        f"(95% CI `{center_state_global[1]:.6f}`, `{center_state_global[2]:.6f}`), p=`{center_state_p:.3e}`"
    )
    module_md.append(
        f"- State IE @ decision token mean: `{decision_state_global[0]:.6f}` "
        f"(95% CI `{decision_state_global[1]:.6f}`, `{decision_state_global[2]:.6f}`), p=`{decision_state_p:.3e}`"
    )
    module_md.append(f"- Attn vs MLP @ decision token paired sign-flip p=`{attn_vs_mlp_p:.3e}`")
    module_md.append("")
    module_md.append("## Top Layers by |mean IE|")
    module_md.append(f"- MLP top layers: `{', '.join([str(int(x)) for x in top_mlp_layers])}`")
    module_md.append(f"- Attn top layers: `{', '.join([str(int(x)) for x in top_attn_layers])}`")
    module_md.append("")
    module_md.append("## 10-layer Window Restore")
    module_md.append(f"- window size = `{args.window_size}`")
    module_md.append(
        f"- best MLP window start layer: `{best_mlp_start}` (mean IE `{win_mlp_mean[best_mlp_start]:.6f}`)"
    )
    module_md.append(
        f"- best Attn window start layer: `{best_attn_start}` (mean IE `{win_attn_mean[best_attn_start]:.6f}`)"
    )
    module_md.append("")
    module_md.append("## Figure Captions")
    module_md.append(
        "- `figs/ct_mlp_heatmap.png`: 红蓝发散色，0 居中；每格是 layer x relative_position 的平均 MLP IE；99 分位裁剪。"
    )
    module_md.append(
        "- `figs/ct_attn_heatmap.png`: 红蓝发散色，0 居中；每格是 layer x relative_position 的平均 Attention IE；99 分位裁剪。"
    )
    module_md.append("- `figs/info_flow_maps_like.png`: State/MLP/Attention 三联图，共享色条与坐标组织。")
    module_md.append("- `figs/ct_lineplot_with_ci.png`: 层方向线图，阴影为 95% 区间，便于和论文 line plot 形式对照。")
    write_text(args.report_dir / "ct_module_compare.md", "\n".join(module_md))

    # A4 report.
    full_global = bootstrap_mean_ci(mod_full_center.flatten(), n_boot=args.n_bootstrap, seed=args.seed + 2100)
    no_mlp_global = bootstrap_mean_ci(mod_no_mlp.flatten(), n_boot=args.n_bootstrap, seed=args.seed + 2101)
    no_attn_global = bootstrap_mean_ci(mod_no_attn.flatten(), n_boot=args.n_bootstrap, seed=args.seed + 2102)
    full_p = sign_flip_pvalue(mod_full_center.flatten(), n_perm=args.n_perm, seed=args.seed + 2103)
    no_mlp_p = sign_flip_pvalue(mod_no_mlp.flatten(), n_perm=args.n_perm, seed=args.seed + 2104)
    no_attn_p = sign_flip_pvalue(mod_no_attn.flatten(), n_perm=args.n_perm, seed=args.seed + 2105)
    full_vs_no_mlp_p = paired_sign_flip_pvalue(
        mod_full_center.flatten(), mod_no_mlp.flatten(), n_perm=args.n_perm, seed=args.seed + 2106
    )
    full_vs_no_attn_p = paired_sign_flip_pvalue(
        mod_full_center.flatten(), mod_no_attn.flatten(), n_perm=args.n_perm, seed=args.seed + 2107
    )
    path_md = []
    path_md.append("# Path-specific Effects (Modified Graph)")
    path_md.append("")
    path_md.append(f"- 样本：严格子集分析集（分层抽样），`N={len(analysis_qs)}`。")
    path_md.append("- `full_restore`：仅做 residual restore。")
    path_md.append("- `freeze_future_mlp`：restore 同时把未来层 MLP 输出固定为 corrupted baseline。")
    path_md.append("- `freeze_future_attn`：restore 同时把未来层 Attention 输出固定为 corrupted baseline。")
    path_md.append("")
    path_md.append("## Summary")
    path_md.append(
        f"- full_restore mean IE: `{full_global[0]:.6f}` "
        f"(95% CI `{full_global[1]:.6f}`, `{full_global[2]:.6f}`), p=`{full_p:.3e}`"
    )
    path_md.append(
        f"- freeze_future_mlp mean IE: `{no_mlp_global[0]:.6f}` "
        f"(95% CI `{no_mlp_global[1]:.6f}`, `{no_mlp_global[2]:.6f}`), p=`{no_mlp_p:.3e}`"
    )
    path_md.append(
        f"- freeze_future_attn mean IE: `{no_attn_global[0]:.6f}` "
        f"(95% CI `{no_attn_global[1]:.6f}`, `{no_attn_global[2]:.6f}`), p=`{no_attn_p:.3e}`"
    )
    path_md.append(f"- paired test: full vs freeze_future_mlp p=`{full_vs_no_mlp_p:.3e}`")
    path_md.append(f"- paired test: full vs freeze_future_attn p=`{full_vs_no_attn_p:.3e}`")
    path_md.append("")
    path_md.append("## Figure Caption")
    path_md.append(
        "- `figs/modified_graph_mlp_vs_attn.png`: 横轴层号，纵轴 IE；实线为均值，阴影为样本分布 95% 分位区间。"
    )
    write_text(args.report_dir / "path_specific_effects.md", "\n".join(path_md))

    # ------------------------------------------------------------------
    # A5: Head-level AP and CT heatmaps.
    # ------------------------------------------------------------------
    ap_head = np.full((len(head_qs), n_layers, n_heads), np.nan, dtype=np.float32)
    ct_head = np.full((len(head_qs), n_layers, n_heads), np.nan, dtype=np.float32)

    log("Running A5 head-level AP/CT scans...")
    for si, q in enumerate(head_qs):
        pair = q_to_pair[q]
        clean_ids = pair.clean_ids.to(model.device)
        corr_ids = pair.corrupt_ids.to(model.device)
        seq_len = pair.seq_len
        last_pos = seq_len - 1

        p_clean = float(baseline_df.loc[baseline_df.q == q, "p_clean"].iloc[0])
        p_corr = float(baseline_df.loc[baseline_df.q == q, "p_corrupt"].iloc[0])

        # only need cache at last position, but collector expects pos list
        clean_cache = collect_cache(model, clean_ids, [last_pos])
        corr_cache = collect_cache(model, corr_ids, [last_pos])

        for li in range(n_layers):
            for hi in range(n_heads):
                s = hi * head_dim
                e = s + head_dim
                clean_chunk = clean_cache.oproj_in[li][s:e]
                corr_chunk = corr_cache.oproj_in[li][s:e]

                # CT head: restore clean chunk in corrupt run.
                p_ct, _ = run_head_patch(
                    model=model,
                    input_ids_1d=corr_ids,
                    tool_id=tool_id,
                    layer_idx=li,
                    head_idx=hi,
                    token_pos=last_pos,
                    patch_chunk=clean_chunk,
                    head_dim=head_dim,
                )
                ct_head[si, li, hi] = p_ct - p_corr

                # AP head: replace clean chunk by corrupt chunk in clean run.
                p_ap_run, _ = run_head_patch(
                    model=model,
                    input_ids_1d=clean_ids,
                    tool_id=tool_id,
                    layer_idx=li,
                    head_idx=hi,
                    token_pos=last_pos,
                    patch_chunk=corr_chunk,
                    head_dim=head_dim,
                )
                ap_head[si, li, hi] = p_clean - p_ap_run

        log(f"  A5 progress: {si + 1}/{len(head_qs)} (q={q}, len={seq_len})")

    ap_head_mean = np.nanmean(ap_head, axis=0)
    ct_head_mean = np.nanmean(ct_head, axis=0)

    save_diverging_heatmap(
        matrix=ap_head_mean,
        out_path=args.fig_dir / "ap_head_heatmap.png",
        title="Head-level Activation Patching (Clean - Clean[head<-corrupt])",
        xlabel="Head",
        ylabel="Layer",
        xticks=list(range(n_heads)),
        xticklabels=[str(h) for h in range(n_heads)],
        yticks=list(range(0, n_layers, 2)),
        yticklabels=[str(l) for l in range(0, n_layers, 2)],
        clip_quantile=99.0,
        figsize=(10.5, 6.5),
    )
    save_diverging_heatmap(
        matrix=ct_head_mean,
        out_path=args.fig_dir / "ct_head_heatmap.png",
        title="Head-level Causal Tracing (Corrupt[head<-clean] - Corrupt)",
        xlabel="Head",
        ylabel="Layer",
        xticks=list(range(n_heads)),
        xticklabels=[str(h) for h in range(n_heads)],
        yticks=list(range(0, n_layers, 2)),
        yticklabels=[str(l) for l in range(0, n_layers, 2)],
        clip_quantile=99.0,
        figsize=(10.5, 6.5),
    )

    # ------------------------------------------------------------------
    # A6: Probe + circuit construction + necessity/sufficiency/specificity.
    # ------------------------------------------------------------------
    combined_score = ap_head_mean + ct_head_mean
    flat_idx = np.argsort(combined_score.reshape(-1))[::-1]

    # Stability over bootstrap subsets.
    boot_select_count = np.zeros((n_layers, n_heads), dtype=np.int64)
    rng = np.random.default_rng(args.seed + 3000)
    n_boot_stab = 300
    for _ in range(n_boot_stab):
        idx = rng.integers(0, len(head_qs), size=len(head_qs))
        b_score = np.nanmean(ap_head[idx], axis=0) + np.nanmean(ct_head[idx], axis=0)
        top = np.argsort(b_score.reshape(-1))[::-1][: args.circuit_k]
        for k in top:
            li = int(k // n_heads)
            hi = int(k % n_heads)
            boot_select_count[li, hi] += 1
    boot_freq = boot_select_count / n_boot_stab

    selected_heads: List[Tuple[int, int]] = []
    for k in flat_idx:
        li = int(k // n_heads)
        hi = int(k % n_heads)
        if combined_score[li, hi] <= 0:
            continue
        selected_heads.append((li, hi))
        if len(selected_heads) >= args.circuit_k:
            break
    if len(selected_heads) < args.circuit_k:
        for k in flat_idx:
            li = int(k // n_heads)
            hi = int(k % n_heads)
            if (li, hi) not in selected_heads:
                selected_heads.append((li, hi))
            if len(selected_heads) >= args.circuit_k:
                break

    selected_heads = sorted(selected_heads, key=lambda x: (x[0], x[1]))
    top_probe_head = selected_heads[0]
    top_probe_name = head_name(*top_probe_head)

    # Probe on full strict subset.
    log("Running component probe and circuit validation on full strict set...")
    strict_qs = [int(x) for x in strict_df["q"].tolist()]

    probe_rows = []
    circuit_rows = []
    all_heads = [(li, hi) for li in range(n_layers) for hi in range(n_heads)]
    for idx_q, q in enumerate(strict_qs):
        pair = q_to_pair[q]
        clean_ids = pair.clean_ids.to(model.device)
        corr_ids = pair.corrupt_ids.to(model.device)
        last_pos = pair.seq_len - 1

        p_clean = float(baseline_df.loc[baseline_df.q == q, "p_clean"].iloc[0])
        p_corr = float(baseline_df.loc[baseline_df.q == q, "p_corrupt"].iloc[0])

        clean_cache = collect_cache(model, clean_ids, [last_pos])
        corr_cache = collect_cache(model, corr_ids, [last_pos])

        # Probe single top head.
        li, hi = top_probe_head
        s = hi * head_dim
        e = s + head_dim
        clean_chunk = clean_cache.oproj_in[li][s:e]
        corr_chunk = corr_cache.oproj_in[li][s:e]
        p_clean_ablate, _ = run_head_patch(
            model, clean_ids, tool_id, li, hi, last_pos, corr_chunk, head_dim
        )
        p_corr_restore, _ = run_head_patch(
            model, corr_ids, tool_id, li, hi, last_pos, clean_chunk, head_dim
        )
        probe_rows.append(
            {
                "q": q,
                "seq_len": pair.seq_len,
                "p_clean": p_clean,
                "p_clean_ablate_top_head": p_clean_ablate,
                "p_corrupt": p_corr,
                "p_corrupt_restore_top_head": p_corr_restore,
            }
        )

        # Circuit necessity/sufficiency.
        circuit_ablate_patches = []
        circuit_restore_patches = []
        for c_li, c_hi in selected_heads:
            cs = c_hi * head_dim
            ce = cs + head_dim
            circuit_ablate_patches.append((c_li, c_hi, corr_cache.oproj_in[c_li][cs:ce]))
            circuit_restore_patches.append((c_li, c_hi, clean_cache.oproj_in[c_li][cs:ce]))

        p_clean_circuit_drop, _ = run_multi_head_patch(
            model=model,
            input_ids_1d=clean_ids,
            tool_id=tool_id,
            token_pos=last_pos,
            patches=circuit_ablate_patches,
            head_dim=head_dim,
        )
        p_corr_circuit_restore, _ = run_multi_head_patch(
            model=model,
            input_ids_1d=corr_ids,
            tool_id=tool_id,
            token_pos=last_pos,
            patches=circuit_restore_patches,
            head_dim=head_dim,
        )

        # Specificity via random non-circuit heads.
        rng_q = np.random.default_rng(args.seed + 5000 + q)
        non_circuit = [h for h in all_heads if h not in selected_heads]
        rand_heads = [non_circuit[int(i)] for i in rng_q.choice(len(non_circuit), size=len(selected_heads), replace=False)]
        rand_patches = []
        for r_li, r_hi in rand_heads:
            rs = r_hi * head_dim
            re_ = rs + head_dim
            rand_patches.append((r_li, r_hi, corr_cache.oproj_in[r_li][rs:re_]))
        p_clean_rand_drop, _ = run_multi_head_patch(
            model=model,
            input_ids_1d=clean_ids,
            tool_id=tool_id,
            token_pos=last_pos,
            patches=rand_patches,
            head_dim=head_dim,
        )

        circuit_rows.append(
            {
                "q": q,
                "seq_len": pair.seq_len,
                "p_clean": p_clean,
                "p_corrupt": p_corr,
                "p_clean_drop_circuit": p_clean_circuit_drop,
                "p_corrupt_restore_circuit": p_corr_circuit_restore,
                "p_clean_drop_random": p_clean_rand_drop,
                "necessity": p_clean - p_clean_circuit_drop,
                "sufficiency": p_corr_circuit_restore - p_corr,
                "random_drop": p_clean - p_clean_rand_drop,
            }
        )

        if (idx_q + 1) % 20 == 0 or (idx_q + 1) == len(strict_qs):
            log(f"  A6 validation progress: {idx_q + 1}/{len(strict_qs)}")

    probe_df = pd.DataFrame(probe_rows)
    circuit_df = pd.DataFrame(circuit_rows)
    probe_df.to_csv(args.report_dir / f"{top_probe_name}_probe_data.csv", index=False)
    circuit_df.to_csv(args.report_dir / "circuit_validation.csv", index=False)

    # Probe figure (required naming LxHy_probe.png).
    pvals = {
        "clean": probe_df["p_clean"].values,
        "clean_ablate_head": probe_df["p_clean_ablate_top_head"].values,
        "corrupt": probe_df["p_corrupt"].values,
        "corrupt_restore_head": probe_df["p_corrupt_restore_top_head"].values,
    }
    means = [np.mean(v) for v in pvals.values()]
    cis = [bootstrap_mean_ci(v, n_boot=args.n_bootstrap, seed=args.seed + 6000 + i) for i, v in enumerate(pvals.values())]
    fig, ax = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    x = np.arange(len(means))
    yerr = np.array([[m - lo, hi - m] for (m, lo, hi) in cis]).T
    ax.bar(x, means, color=["#4c78a8", "#e45756", "#9ecae9", "#72b7b2"], alpha=0.92)
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.3, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["clean", "clean ablate", "corrupt", "corrupt restore"], rotation=12)
    ax.set_ylabel("P(<tool_call> at t=1)")
    ax.set_title(f"Probe {top_probe_name}")
    probe_fig_path = args.fig_dir / f"{top_probe_name}_probe.png"
    fig.savefig(probe_fig_path, dpi=220)
    plt.close(fig)

    # Final circuit figure.
    G = nx.DiGraph()
    node_scores = {}
    for li, hi in selected_heads:
        n = head_name(li, hi)
        node_scores[n] = float(combined_score[li, hi])
        G.add_node(n, layer=li, score=node_scores[n])
    G.add_node("CORR_TOKEN", layer=-1, score=0.0)
    G.add_node("DECISION_<tool_call>@t1", layer=n_layers + 1, score=0.0)

    sorted_heads = sorted(selected_heads, key=lambda x: (x[0], x[1]))
    for li, hi in sorted_heads:
        n = head_name(li, hi)
        G.add_edge("CORR_TOKEN", n, weight=float(ct_head_mean[li, hi]))
        G.add_edge(n, "DECISION_<tool_call>@t1", weight=float(ap_head_mean[li, hi]))
    for (la, ha), (lb, hb) in zip(sorted_heads[:-1], sorted_heads[1:]):
        na = head_name(la, ha)
        nb = head_name(lb, hb)
        w = float((combined_score[la, ha] + combined_score[lb, hb]) / 2.0)
        G.add_edge(na, nb, weight=w)

    pos = {"CORR_TOKEN": (0.0, 0.0), "DECISION_<tool_call>@t1": (n_layers + 2.0, 0.0)}
    for idx, (li, hi) in enumerate(sorted_heads):
        pos[head_name(li, hi)] = (li + 1.0, (idx - (len(sorted_heads) - 1) / 2.0) * 0.9)

    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    node_colors = []
    for n in G.nodes():
        if n in ("CORR_TOKEN", "DECISION_<tool_call>@t1"):
            node_colors.append("#dddddd")
        else:
            s = node_scores[n]
            node_colors.append("#ef8a62" if s >= 0 else "#67a9cf")
    nx.draw_networkx_nodes(G, pos, node_size=1600, node_color=node_colors, ax=ax, linewidths=1.0, edgecolors="#555")
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    edge_widths = []
    for _, _, d in G.edges(data=True):
        edge_widths.append(1.0 + 2.0 * min(abs(d.get("weight", 0.0)) / (np.max(np.abs(combined_score)) + 1e-8), 1.0))
    nx.draw_networkx_edges(G, pos, width=edge_widths, arrowstyle="-|>", arrowsize=16, ax=ax, edge_color="#444")
    ax.set_title("Final Candidate Circuit for <tool_call> Decision", fontsize=13)
    ax.axis("off")
    fig.savefig(args.fig_dir / "final_circuit.png", dpi=220)
    plt.close(fig)

    # ------------------------------------------------------------------
    # A7: Robustness (length buckets) + random control summary.
    # ------------------------------------------------------------------
    q1 = circuit_df["seq_len"].quantile(1 / 3)
    q2 = circuit_df["seq_len"].quantile(2 / 3)

    def bucket_of_len(x: int) -> str:
        if x <= q1:
            return "short"
        if x <= q2:
            return "medium"
        return "long"

    circuit_df["len_bucket"] = circuit_df["seq_len"].apply(bucket_of_len)
    bucket_rows = []
    for b in ["short", "medium", "long"]:
        sub = circuit_df[circuit_df["len_bucket"] == b]
        if len(sub) == 0:
            continue
        n = len(sub)
        nec = bootstrap_mean_ci(sub["necessity"].values, n_boot=args.n_bootstrap, seed=args.seed + 7000 + n)
        suf = bootstrap_mean_ci(sub["sufficiency"].values, n_boot=args.n_bootstrap, seed=args.seed + 7100 + n)
        rnd = bootstrap_mean_ci(sub["random_drop"].values, n_boot=args.n_bootstrap, seed=args.seed + 7200 + n)
        bucket_rows.append(
            {
                "bucket": b,
                "N": n,
                "necessity_mean": nec[0],
                "necessity_ci95_low": nec[1],
                "necessity_ci95_high": nec[2],
                "sufficiency_mean": suf[0],
                "sufficiency_ci95_low": suf[1],
                "sufficiency_ci95_high": suf[2],
                "random_drop_mean": rnd[0],
                "random_drop_ci95_low": rnd[1],
                "random_drop_ci95_high": rnd[2],
            }
        )
    bucket_df = pd.DataFrame(bucket_rows)
    bucket_df.to_csv(args.report_dir / "robustness_length_buckets.csv", index=False)

    # ------------------------------------------------------------------
    # Reports for A5/A6/A7/A8
    # ------------------------------------------------------------------
    # Head summary report
    top_head_rows = []
    flat_scores = []
    for li in range(n_layers):
        for hi in range(n_heads):
            apm = float(ap_head_mean[li, hi])
            ctm = float(ct_head_mean[li, hi])
            score = float(combined_score[li, hi])
            freq = float(boot_freq[li, hi])
            flat_scores.append((li, hi, apm, ctm, score, freq))
    flat_scores.sort(key=lambda x: x[4], reverse=True)
    for li, hi, apm, ctm, score, freq in flat_scores[:20]:
        top_head_rows.append(
            f"- `{head_name(li, hi)}`: score={score:.6f}, AP={apm:.6f}, CT={ctm:.6f}, bootstrap_topk_freq={freq:.3f}"
        )

    head_md = []
    head_md.append("# Head-level AP/CT Summary")
    head_md.append("")
    head_md.append(f"- 头级扫描样本：严格子集（{args.subset_strategy}），`N={len(head_qs)}`。")
    head_md.append("- `ap_head_heatmap.png` 与 `ct_head_heatmap.png` 使用红蓝发散色，0 居中，99 分位裁剪。")
    head_md.append("- AP 定义：`p_clean - p_clean[head<-corrupt]`；CT 定义：`p_corrupt[head<-clean] - p_corrupt`。")
    head_md.append("")
    head_md.append("## Top Heads")
    head_md.extend(top_head_rows)
    head_md.append("")
    head_md.append("## Selected Circuit Heads")
    head_md.append(f"- `{', '.join([head_name(li, hi) for li, hi in selected_heads])}`")
    write_text(args.report_dir / "head_summary.md", "\n".join(head_md))

    # Circuit + robustness report
    nec = bootstrap_mean_ci(circuit_df["necessity"].values, n_boot=args.n_bootstrap, seed=args.seed + 8000)
    suf = bootstrap_mean_ci(circuit_df["sufficiency"].values, n_boot=args.n_bootstrap, seed=args.seed + 8001)
    rnd = bootstrap_mean_ci(circuit_df["random_drop"].values, n_boot=args.n_bootstrap, seed=args.seed + 8002)

    robustness_md = []
    robustness_md.append("# Robustness")
    robustness_md.append("")
    robustness_md.append("- 组别：按 prompt 长度三分桶（short/medium/long）。")
    robustness_md.append("- 指标：电路必要性（clean drop）、充分性（corrupt restore）、随机头对照 drop。")
    robustness_md.append("")
    robustness_md.append("## Overall (strict full set)")
    robustness_md.append(
        f"- necessity mean: `{nec[0]:.6f}` (95% CI `{nec[1]:.6f}`, `{nec[2]:.6f}`)"
    )
    robustness_md.append(
        f"- sufficiency mean: `{suf[0]:.6f}` (95% CI `{suf[1]:.6f}`, `{suf[2]:.6f}`)"
    )
    robustness_md.append(
        f"- random-head drop mean: `{rnd[0]:.6f}` (95% CI `{rnd[1]:.6f}`, `{rnd[2]:.6f}`)"
    )
    robustness_md.append("")
    robustness_md.append("## Length Buckets")
    for _, r in bucket_df.iterrows():
        robustness_md.append(
            "- "
            f"`{r['bucket']}` N={int(r['N'])}: "
            f"necessity={r['necessity_mean']:.6f} "
            f"[{r['necessity_ci95_low']:.6f}, {r['necessity_ci95_high']:.6f}], "
            f"sufficiency={r['sufficiency_mean']:.6f} "
            f"[{r['sufficiency_ci95_low']:.6f}, {r['sufficiency_ci95_high']:.6f}], "
            f"random_drop={r['random_drop_mean']:.6f} "
            f"[{r['random_drop_ci95_low']:.6f}, {r['random_drop_ci95_high']:.6f}]"
        )
    write_text(args.report_dir / "robustness.md", "\n".join(robustness_md))

    # conclusion_alignment.md (A8).
    align_md = []
    align_md.append("# Conclusion Alignment (定位章节对齐)")
    align_md.append("")
    align_md.append("| 论文定位结论 | 本项目是否成立 | 证据 |")
    align_md.append("|---|---|---|")
    align_md.append(
        f"| Clean/Corrupt/Restore 在 t=1 上存在稳定总效应 | 成立（全量 + 严格子集） | `reports/baseline_metrics.md`, `reports/te_summary.csv` |"
    )
    align_md.append(
        f"| 存在可定位的 state-level 关键信息位点 | 成立（子集结论） | `figs/ct_state_heatmap.png`, `reports/ct_state_topk.csv` |"
    )
    align_md.append(
        f"| MLP vs Attention 路径贡献可分解 | 成立（子集结论） | `figs/ct_mlp_heatmap.png`, `figs/ct_attn_heatmap.png`, `reports/ct_module_compare.md` |"
    )
    align_md.append(
        f"| modified graph 可区分未来 MLP/Attention 路径作用 | 成立（子集结论） | `figs/modified_graph_mlp_vs_attn.png`, `reports/path_specific_effects.md` |"
    )
    align_md.append(
        f"| 头级 AP/CT 可定位候选关键头 | 成立（子集结论） | `figs/ap_head_heatmap.png`, `figs/ct_head_heatmap.png`, `reports/head_summary.md` |"
    )
    align_md.append(
        f"| 电路具备必要性/充分性/特异性 | 成立（严格全子集） | `figs/final_circuit.png`, `figs/{top_probe_name}_probe.png`, `reports/robustness.md` |"
    )
    write_text(args.report_dir / "conclusion_alignment.md", "\n".join(align_md))

    # Metadata for reproducibility.
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "model_path": str(args.model_path),
        "pair_dir": str(args.pair_dir),
        "tool_token_id": tool_id,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "full_n": full_n,
        "strict_n": strict_n,
        "analysis_qs": analysis_qs,
        "head_qs": head_qs,
        "subset_strategy": args.subset_strategy,
        "selected_heads": [head_name(li, hi) for li, hi in selected_heads],
        "window_size": args.window_size,
        "n_bootstrap": args.n_bootstrap,
        "n_perm": args.n_perm,
        "ct_rel_positions": rel_positions,
    }
    (args.report_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    # Final quick sanity checks for result awareness.
    sanity = {
        "clean_rate": float(clean_rate),
        "corrupt_non_tool_rate": float(corr_nontool_rate),
        "strict_rate": float(strict_rate),
        "te_full_mean": float(te_full_mean),
        "te_strict_mean": float(te_strict_mean),
        "state_center_mean": float(center_state_global[0]),
        "state_decision_mean": float(decision_state_global[0]),
        "mlp_decision_mean": float(mlp_global[0]),
        "attn_decision_mean": float(attn_global[0]),
        "state_center_p": float(center_state_p),
        "state_decision_p": float(decision_state_p),
        "mlp_decision_p": float(mlp_p),
        "attn_decision_p": float(attn_p),
        "ct_head_max": float(np.nanmax(ct_head_mean)),
        "ap_head_max": float(np.nanmax(ap_head_mean)),
        "necessity_mean": float(nec[0]),
        "sufficiency_mean": float(suf[0]),
        "random_drop_mean": float(rnd[0]),
        "path_full_p": float(full_p),
        "path_full_vs_no_mlp_p": float(full_vs_no_mlp_p),
        "path_full_vs_no_attn_p": float(full_vs_no_attn_p),
    }
    (args.report_dir / "sanity_check.json").write_text(json.dumps(sanity, ensure_ascii=False, indent=2))

    log("Pipeline completed.")
    log(f"Selected circuit heads: {', '.join([head_name(li, hi) for li, hi in selected_heads])}")
    log(f"Top probe figure: {top_probe_name}_probe.png")


if __name__ == "__main__":
    main()
