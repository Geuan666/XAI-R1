#!/usr/bin/env python3
"""Paper-style causal tracing alignment run.

Goal:
- Reorganize CT/AP-related plots to match "Interpretability in the Wild" style:
  state / MLP / Attn AIE maps with token-position x layer axes, plus lineplots + CI.
- Strengthen statistical evidence with larger strict subset and explicit significance tests.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import wilcoxon
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_interpretability_pipeline import (
    bootstrap_mean_ci,
    load_pairs,
    run_prob,
    run_resid_restore,
    run_multi_attn_restore,
    run_multi_mlp_restore,
    set_seed,
)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


@dataclass
class CacheMulti:
    resid_pre: List[torch.Tensor]  # [L], each [P, D]
    attn_out: List[torch.Tensor]  # [L], each [P, D]
    mlp_out: List[torch.Tensor]  # [L], each [P, D]


def collect_cache_multi(
    model,
    input_ids_1d: torch.Tensor,
    pos_list: Sequence[int],
) -> CacheMulti:
    layers = model.model.layers
    n_layers = len(layers)
    device = input_ids_1d.device
    pos_tensor = torch.tensor(pos_list, device=device, dtype=torch.long)

    resid_pre: List[torch.Tensor] = [None] * n_layers
    attn_out: List[torch.Tensor] = [None] * n_layers
    mlp_out: List[torch.Tensor] = [None] * n_layers
    handles = []

    for li, layer in enumerate(layers):
        def make_pre(idx: int):
            def _hook(module, inputs):
                hs = inputs[0]  # [1, seq, d]
                resid_pre[idx] = hs[0].index_select(0, pos_tensor).detach().clone()
            return _hook

        def make_attn(idx: int):
            def _hook(module, inputs, output):
                out = output[0] if isinstance(output, tuple) else output
                attn_out[idx] = out[0].index_select(0, pos_tensor).detach().clone()
            return _hook

        def make_mlp(idx: int):
            def _hook(module, inputs, output):
                mlp_out[idx] = output[0].index_select(0, pos_tensor).detach().clone()
            return _hook

        handles.append(layer.register_forward_pre_hook(make_pre(li)))
        handles.append(layer.self_attn.register_forward_hook(make_attn(li)))
        handles.append(layer.mlp.register_forward_hook(make_mlp(li)))

    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids_1d.unsqueeze(0), use_cache=False)
    finally:
        for h in handles:
            h.remove()

    for li in range(n_layers):
        if resid_pre[li] is None or attn_out[li] is None or mlp_out[li] is None:
            raise RuntimeError(f"Cache missing at layer {li}")

    return CacheMulti(resid_pre=resid_pre, attn_out=attn_out, mlp_out=mlp_out)


def build_diverging_norm(mat: np.ndarray, q: float = 99.0) -> TwoSlopeNorm:
    vals = np.abs(mat[np.isfinite(mat)])
    v = float(np.quantile(vals, q / 100.0)) if vals.size > 0 else 1.0
    v = max(v, 1e-8)
    return TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)


def save_triptych(
    state_mat: np.ndarray,
    mlp_mat: np.ndarray,
    attn_mat: np.ndarray,
    rel_positions: Sequence[int],
    out_path: Path,
    title: str,
) -> None:
    mats = [state_mat, mlp_mat, attn_mat]
    sub_titles = [
        "State AIE (Residual Restore)",
        "MLP AIE (10-layer Window Restore)",
        "Attention AIE (10-layer Window Restore)",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for ax, mat, st in zip(axes, mats, sub_titles):
        norm = build_diverging_norm(mat, q=99.0)
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
        )
        ax.set_title(st, fontsize=11)
        ax.set_xlabel("Relative Position")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(len(rel_positions)))
        ax.set_xticklabels([str(x) for x in rel_positions], rotation=45, fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("AIE", fontsize=9)
    fig.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def save_single_heatmap(
    mat: np.ndarray,
    rel_positions: Sequence[int],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    norm = build_diverging_norm(mat, q=99.0)
    im = ax.imshow(
        mat,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Relative Position")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(rel_positions)))
    ax.set_xticklabels([str(x) for x in rel_positions])
    ax.set_yticks(range(0, mat.shape[0], 2))
    ax.set_yticklabels([str(x) for x in range(0, mat.shape[0], 2)])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("AIE on P(<tool_call> at t=1)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def ci_profile(sample_layer: np.ndarray, n_boot: int, seed_base: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sample_layer shape [N, L] -> mean/lo/hi over layer."""
    n_layers = sample_layer.shape[1]
    mean = np.zeros(n_layers, dtype=np.float64)
    lo = np.zeros(n_layers, dtype=np.float64)
    hi = np.zeros(n_layers, dtype=np.float64)
    for l in range(n_layers):
        m, a, b = bootstrap_mean_ci(sample_layer[:, l], n_boot=n_boot, seed=seed_base + l)
        mean[l], lo[l], hi[l] = m, a, b
    return mean, lo, hi


def save_lineplots_with_ci(
    rel0_state: np.ndarray,
    rel0_mlp: np.ndarray,
    rel0_attn: np.ndarray,
    dec_mlp: np.ndarray,
    dec_attn: np.ndarray,
    out_path: Path,
    n_boot: int,
    seed: int,
) -> None:
    layers = np.arange(rel0_state.shape[1])
    st_m, st_lo, st_hi = ci_profile(rel0_state, n_boot, seed + 10)
    m0_m, m0_lo, m0_hi = ci_profile(rel0_mlp, n_boot, seed + 100)
    a0_m, a0_lo, a0_hi = ci_profile(rel0_attn, n_boot, seed + 200)
    md_m, md_lo, md_hi = ci_profile(dec_mlp, n_boot, seed + 300)
    ad_m, ad_lo, ad_hi = ci_profile(dec_attn, n_boot, seed + 400)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)

    # (a) state trace at rel=0
    ax = axes[0]
    ax.plot(layers, st_m, color="#7f3b08", lw=2.0, label="State (rel=0)")
    ax.fill_between(layers, st_lo, st_hi, color="#7f3b08", alpha=0.2)
    ax.axhline(0.0, color="gray", ls="--", lw=1.0)
    ax.set_title("State AIE Profile (rel=0)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AIE")
    ax.legend(frameon=False, fontsize=9)

    # (b) rel=0 mlp vs attn
    ax = axes[1]
    ax.plot(layers, m0_m, color="#d62728", lw=2.0, label="MLP (rel=0)")
    ax.fill_between(layers, m0_lo, m0_hi, color="#d62728", alpha=0.16)
    ax.plot(layers, a0_m, color="#1f77b4", lw=2.0, label="Attn (rel=0)")
    ax.fill_between(layers, a0_lo, a0_hi, color="#1f77b4", alpha=0.16)
    ax.axhline(0.0, color="gray", ls="--", lw=1.0)
    ax.set_title("Early Site: MLP vs Attention")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AIE")
    ax.legend(frameon=False, fontsize=9)

    # (c) decision position mlp vs attn
    ax = axes[2]
    ax.plot(layers, md_m, color="#d62728", lw=2.0, label="MLP (decision)")
    ax.fill_between(layers, md_lo, md_hi, color="#d62728", alpha=0.16)
    ax.plot(layers, ad_m, color="#1f77b4", lw=2.0, label="Attn (decision)")
    ax.fill_between(layers, ad_lo, ad_hi, color="#1f77b4", alpha=0.16)
    ax.axhline(0.0, color="gray", ls="--", lw=1.0)
    ax.set_title("Late Site: MLP vs Attention")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AIE")
    ax.legend(frameon=False, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/root/data/Qwen/Qwen3-1.7B"))
    parser.add_argument("--pair-dir", type=Path, default=Path("pair"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--rel-window", type=int, default=16)
    parser.add_argument("--rel-stride", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--max-strict", type=int, default=0, help="0 means all strict samples")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    log("Loading model/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    tool_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    if len(tool_ids) != 1:
        raise RuntimeError(f"Unexpected <tool_call> tokenization: {tool_ids}")
    tool_id = int(tool_ids[0])

    pairs = load_pairs(args.pair_dir, tokenizer)
    if not pairs:
        raise RuntimeError("No aligned pairs found.")
    q2pair = {p.q: p for p in pairs}

    # Baseline to get strict set + TE.
    log("Computing strict set baseline...")
    rows = []
    for p in pairs:
        clean_ids = p.clean_ids.to(model.device)
        corr_ids = p.corrupt_ids.to(model.device)
        p_clean, top_clean = run_prob(model, clean_ids, tool_id)
        p_corr, top_corr = run_prob(model, corr_ids, tool_id)
        rows.append(
            {
                "q": p.q,
                "seq_len": p.seq_len,
                "p_clean": p_clean,
                "p_corrupt": p_corr,
                "TE": p_clean - p_corr,
                "strict": int(top_clean == tool_id and top_corr != tool_id),
            }
        )
    bdf = pd.DataFrame(rows).sort_values("q").reset_index(drop=True)
    strict_df = bdf[bdf["strict"] == 1].copy()
    strict_qs = strict_df["q"].tolist()
    if args.max_strict > 0:
        strict_qs = strict_qs[: args.max_strict]
    if not strict_qs:
        raise RuntimeError("Strict subset empty.")
    log(f"Using strict subset N={len(strict_qs)}")

    n_layers = len(model.model.layers)
    rel_positions = list(range(-args.rel_window, args.rel_window + 1, args.rel_stride))
    n_rel = len(rel_positions)
    rel0_idx = rel_positions.index(0)

    # Store per-sample tensors.
    state_ie = np.full((len(strict_qs), n_layers, n_rel), np.nan, dtype=np.float32)
    mlp_ie = np.full((len(strict_qs), n_layers, n_rel), np.nan, dtype=np.float32)
    attn_ie = np.full((len(strict_qs), n_layers, n_rel), np.nan, dtype=np.float32)
    state_nie = np.full((len(strict_qs), n_layers, n_rel), np.nan, dtype=np.float32)
    mlp_nie = np.full((len(strict_qs), n_layers, n_rel), np.nan, dtype=np.float32)
    attn_nie = np.full((len(strict_qs), n_layers, n_rel), np.nan, dtype=np.float32)
    dec_mlp_ie = np.full((len(strict_qs), n_layers), np.nan, dtype=np.float32)
    dec_attn_ie = np.full((len(strict_qs), n_layers), np.nan, dtype=np.float32)

    log("Running paper-style AIE scans (state + MLP/Attn windows)...")
    for si, q in enumerate(strict_qs, start=1):
        pair = q2pair[int(q)]
        clean_ids = pair.clean_ids.to(model.device)
        corr_ids = pair.corrupt_ids.to(model.device)
        seq_len = pair.seq_len

        p_clean = float(strict_df.loc[strict_df.q == q, "p_clean"].iloc[0])
        p_corr = float(strict_df.loc[strict_df.q == q, "p_corrupt"].iloc[0])
        te = max(p_clean - p_corr, 1e-6)

        diff = pair.diff_positions
        center = diff[len(diff) // 2] if diff else max(0, seq_len - 4)
        abs_rel = [min(max(center + r, 0), seq_len - 1) for r in rel_positions]
        decision_pos = seq_len - 1
        all_pos = abs_rel + [decision_pos]

        clean_cache = collect_cache_multi(model, clean_ids, all_pos)

        # position index helpers
        rel_to_cache_idx = {ri: ri for ri in range(n_rel)}
        dec_idx = len(all_pos) - 1

        # state-level IE
        for li in range(n_layers):
            for ri, pos in enumerate(abs_rel):
                p_patch, _ = run_resid_restore(
                    model=model,
                    input_ids_1d=corr_ids,
                    tool_id=tool_id,
                    layer_idx=li,
                    token_pos=pos,
                    patch_vec=clean_cache.resid_pre[li][rel_to_cache_idx[ri]],
                )
                ie = p_patch - p_corr
                state_ie[si - 1, li, ri] = ie
                state_nie[si - 1, li, ri] = ie / te

        # window restore maps for MLP/Attn on relative positions.
        for li in range(n_layers):
            l0 = max(0, li - 4)
            l1 = min(n_layers, li + 6)
            win_layers = list(range(l0, l1))
            for ri, pos in enumerate(abs_rel):
                mlp_dict = {l: clean_cache.mlp_out[l][rel_to_cache_idx[ri]] for l in win_layers}
                attn_dict = {l: clean_cache.attn_out[l][rel_to_cache_idx[ri]] for l in win_layers}

                p_m, _ = run_multi_mlp_restore(model, corr_ids, tool_id, mlp_dict, pos)
                p_a, _ = run_multi_attn_restore(model, corr_ids, tool_id, attn_dict, pos)
                ie_m = p_m - p_corr
                ie_a = p_a - p_corr
                mlp_ie[si - 1, li, ri] = ie_m
                attn_ie[si - 1, li, ri] = ie_a
                mlp_nie[si - 1, li, ri] = ie_m / te
                attn_nie[si - 1, li, ri] = ie_a / te

            # decision-site profiles
            mlp_dict_d = {l: clean_cache.mlp_out[l][dec_idx] for l in win_layers}
            attn_dict_d = {l: clean_cache.attn_out[l][dec_idx] for l in win_layers}
            p_md, _ = run_multi_mlp_restore(model, corr_ids, tool_id, mlp_dict_d, decision_pos)
            p_ad, _ = run_multi_attn_restore(model, corr_ids, tool_id, attn_dict_d, decision_pos)
            dec_mlp_ie[si - 1, li] = p_md - p_corr
            dec_attn_ie[si - 1, li] = p_ad - p_corr

        if si % 5 == 0 or si == len(strict_qs):
            log(f"  progress {si}/{len(strict_qs)}")

    # Aggregate maps.
    state_aie = np.nanmean(state_ie, axis=0)
    mlp_aie = np.nanmean(mlp_ie, axis=0)
    attn_aie = np.nanmean(attn_ie, axis=0)
    state_anie = np.nanmean(state_nie, axis=0)
    mlp_anie = np.nanmean(mlp_nie, axis=0)
    attn_anie = np.nanmean(attn_nie, axis=0)

    # Overwrite required figure names with paper-style organization.
    save_single_heatmap(
        state_aie,
        rel_positions,
        args.fig_dir / "ct_state_heatmap.png",
        "State-level AIE Map (Paper-style)",
    )
    save_single_heatmap(
        mlp_aie,
        rel_positions,
        args.fig_dir / "ct_mlp_heatmap.png",
        "MLP AIE Map (10-layer Window Restore, Paper-style)",
    )
    save_single_heatmap(
        attn_aie,
        rel_positions,
        args.fig_dir / "ct_attn_heatmap.png",
        "Attention AIE Map (10-layer Window Restore, Paper-style)",
    )

    save_triptych(
        state_aie,
        mlp_aie,
        attn_aie,
        rel_positions,
        args.fig_dir / "avg_trace_triptych.png",
        "Average Causal Traces (State / MLP / Attention)",
    )

    # Lineplots with CI.
    rel0_state = state_ie[:, :, rel0_idx]
    rel0_mlp = mlp_ie[:, :, rel0_idx]
    rel0_attn = attn_ie[:, :, rel0_idx]
    save_lineplots_with_ci(
        rel0_state=rel0_state,
        rel0_mlp=rel0_mlp,
        rel0_attn=rel0_attn,
        dec_mlp=dec_mlp_ie,
        dec_attn=dec_attn_ie,
        out_path=args.fig_dir / "ct_lineplot_ci.png",
        n_boot=args.n_bootstrap,
        seed=args.seed,
    )

    # Peak significance analyses.
    # Early site: rel=0, middle layers.
    mid_layers = np.arange(max(0, n_layers // 3), min(n_layers, (2 * n_layers) // 3 + 2))
    low_layers = np.arange(0, max(1, n_layers // 5))
    high_layers = np.arange(max(1, n_layers - n_layers // 5), n_layers)

    mean_mlp_rel0 = rel0_mlp.mean(axis=0)
    early_peak_layer = int(mid_layers[np.argmax(mean_mlp_rel0[mid_layers])])
    early_peak_vals = rel0_mlp[:, early_peak_layer]
    early_bg_vals = np.concatenate([rel0_mlp[:, low_layers], rel0_mlp[:, high_layers]], axis=1).mean(axis=1)
    early_diff = early_peak_vals - early_bg_vals
    early_w = wilcoxon(early_diff, alternative="greater", zero_method="wilcox")
    early_ci = bootstrap_mean_ci(early_diff, n_boot=args.n_bootstrap, seed=args.seed + 10000)

    # Late site: decision token, high-layer attention peak.
    mean_dec_attn = dec_attn_ie.mean(axis=0)
    late_layers = np.arange(max(0, (2 * n_layers) // 3), n_layers)
    late_peak_layer = int(late_layers[np.argmax(mean_dec_attn[late_layers])])
    late_peak_vals = dec_attn_ie[:, late_peak_layer]
    late_bg_vals = dec_attn_ie[:, low_layers].mean(axis=1)
    late_diff = late_peak_vals - late_bg_vals
    late_w = wilcoxon(late_diff, alternative="greater", zero_method="wilcox")
    late_ci = bootstrap_mean_ci(late_diff, n_boot=args.n_bootstrap, seed=args.seed + 11000)

    # MLP vs Attn comparisons at two sites.
    mlp_vs_attn_early = rel0_mlp[:, early_peak_layer] - rel0_attn[:, early_peak_layer]
    mvse_w = wilcoxon(mlp_vs_attn_early, alternative="greater", zero_method="wilcox")
    mvse_ci = bootstrap_mean_ci(mlp_vs_attn_early, n_boot=args.n_bootstrap, seed=args.seed + 12000)

    attn_vs_mlp_late = dec_attn_ie[:, late_peak_layer] - dec_mlp_ie[:, late_peak_layer]
    avml_w = wilcoxon(attn_vs_mlp_late, alternative="greater", zero_method="wilcox")
    avml_ci = bootstrap_mean_ci(attn_vs_mlp_late, n_boot=args.n_bootstrap, seed=args.seed + 13000)

    # Save trace tensors for auditability.
    np.save(args.report_dir / "paper_state_aie.npy", state_aie)
    np.save(args.report_dir / "paper_mlp_aie.npy", mlp_aie)
    np.save(args.report_dir / "paper_attn_aie.npy", attn_aie)
    np.save(args.report_dir / "paper_state_anie.npy", state_anie)
    np.save(args.report_dir / "paper_mlp_anie.npy", mlp_anie)
    np.save(args.report_dir / "paper_attn_anie.npy", attn_anie)

    # Report.
    md = []
    md.append("# Paper-style Alignment and Significance")
    md.append("")
    md.append("## Setup")
    md.append(f"- 模型：`{args.model_path}`")
    md.append(f"- 严格子集样本数：`N={len(strict_qs)}`")
    md.append(f"- 位置网格：`rel_positions={rel_positions}`（相对 corruption 中心）")
    md.append("- 模块恢复窗口：`[l-4, ..., l+5]`（边界裁剪），与论文 10 层窗口一致。")
    md.append("- 主指标：`p_tool = P(<tool_call> at t=1)`；报告 `AIE` 与 `ANIE (= IE/TE)`。")
    md.append("")
    md.append("## Paper-style Key Results")
    md.append(
        f"- Early-site (rel=0) MLP peak layer: `L{early_peak_layer}`; "
        f"peak-vs-background mean diff=`{early_ci[0]:.6f}` "
        f"(95% CI `{early_ci[1]:.6f}`, `{early_ci[2]:.6f}`), "
        f"Wilcoxon one-sided `p={early_w.pvalue:.3e}`."
    )
    md.append(
        f"- Late-site (decision token) Attention peak layer: `L{late_peak_layer}`; "
        f"peak-vs-low-layer mean diff=`{late_ci[0]:.6f}` "
        f"(95% CI `{late_ci[1]:.6f}`, `{late_ci[2]:.6f}`), "
        f"Wilcoxon one-sided `p={late_w.pvalue:.3e}`."
    )
    md.append(
        f"- Early-site module contrast (MLP - Attn @ L{early_peak_layer}, rel=0): "
        f"mean diff=`{mvse_ci[0]:.6f}` (95% CI `{mvse_ci[1]:.6f}`, `{mvse_ci[2]:.6f}`), "
        f"`p={mvse_w.pvalue:.3e}`."
    )
    md.append(
        f"- Late-site module contrast (Attn - MLP @ L{late_peak_layer}, decision): "
        f"mean diff=`{avml_ci[0]:.6f}` (95% CI `{avml_ci[1]:.6f}`, `{avml_ci[2]:.6f}`), "
        f"`p={avml_w.pvalue:.3e}`."
    )
    md.append("")
    md.append("## Output Figures")
    md.append("- `figs/ct_state_heatmap.png` (paper-style state AIE map)")
    md.append("- `figs/ct_mlp_heatmap.png` (paper-style MLP window AIE map)")
    md.append("- `figs/ct_attn_heatmap.png` (paper-style attention window AIE map)")
    md.append("- `figs/avg_trace_triptych.png` (state/MLP/Attn triptych)")
    md.append("- `figs/ct_lineplot_ci.png` (line plots + 95% CI)")
    md.append("")
    md.append("## Figure Caption Notes")
    md.append("- 所有热图采用 `RdBu` 发散配色，0 居中；颜色范围按 |AIE| 的 99 分位裁剪。")
    md.append("- 线图阴影为 bootstrap 95% CI。")
    md.append("- 统计显著性通过 paired Wilcoxon 单侧检验（`alternative='greater'`）报告。")
    (args.report_dir / "paper_style_significance.md").write_text("\n".join(md))

    meta = {
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "strict_n": len(strict_qs),
        "strict_qs": strict_qs,
        "rel_positions": rel_positions,
        "window_size": args.window_size,
        "early_peak_layer": early_peak_layer,
        "late_peak_layer": late_peak_layer,
        "wilcoxon_p": {
            "early_peak_vs_bg": float(early_w.pvalue),
            "late_peak_vs_bg": float(late_w.pvalue),
            "mlp_vs_attn_early": float(mvse_w.pvalue),
            "attn_vs_mlp_late": float(avml_w.pvalue),
        },
    }
    (args.report_dir / "paper_style_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    log("Paper-style alignment run completed.")


if __name__ == "__main__":
    main()
