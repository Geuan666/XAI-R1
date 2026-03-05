# State-level Causal Trace Summary

- 样本：严格子集全量 `N=116`。
- 指标：`IE(layer, pos) = p_tool(corrupt + restore resid_pre_from_clean) - p_tool(corrupt)`。
- 扫描网格：`Layer=0..27`，`rel_pos in [-16, -12, -8, -4, 0, 4, 8, 12, 16]`。

## Key Findings
- AIE 峰值位点：`layer=1, rel_pos=0`，mean IE=`0.836515`。
- 峰值显著性与 layer-profile 置信区间见 `reports/paper_style_significance.md`。
- 当前任务在 rel=0 处呈现更强的早层注意力耦合，与论文事实回忆任务的 MLP 早站点模式不同。

## Figure Caption
- `figs/ct_state_heatmap.png`：RdBu 发散配色，0 居中；颜色范围按 |AIE| 99 分位裁剪；每格为跨样本平均 IE。