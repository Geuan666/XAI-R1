# MLP vs Attention Decomposition

- 口径：论文对齐版（AIE map + lineplot CI），严格子集 `N=116`。
- 指标：`IE = p_tool(corrupt + restore_from_clean) - p_tool(corrupt)`；模块恢复使用 10 层窗口。
- 热图轴：`Layer x Relative Position`（相对 corruption 中心）。

## Peak Statistics (AIE map)
- State map peak: `L1, rel=0`, AIE=`0.836515`
- MLP map peak: `L0, rel=0`, AIE=`0.415706`
- Attention map peak: `L2, rel=0`, AIE=`0.875513`

## rel=0 (Corruption Token) Compare
- mean AIE over layers: MLP=`0.099675`, Attention=`0.224306`
- 数据驱动结论：early site 上 Attention 强于 MLP（显著性见 `reports/paper_style_significance.md`）。

## Figures
- `figs/ct_mlp_heatmap.png`: MLP window-restore AIE map（论文同构组织）。
- `figs/ct_attn_heatmap.png`: Attention window-restore AIE map（论文同构组织）。
- `figs/avg_trace_triptych.png`: State/MLP/Attention 三联图。
- `figs/ct_lineplot_ci.png`: 线图 + 95% CI（对应论文 appendix lineplot 风格）。