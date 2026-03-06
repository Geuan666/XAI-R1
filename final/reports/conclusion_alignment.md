# Conclusion Alignment (定位章节对齐)

| 论文定位结论 | 本项目是否成立 | 证据 |
|---|---|---|
| Clean/Corrupt/Restore 在 t=1 上存在稳定总效应 | 成立（全量 + 严格子集） | `reports/baseline_metrics.md`, `reports/te_summary.csv` |
| 存在可定位的 state-level 关键信息位点 | 成立（严格全子集） | `figs/ct_state_heatmap.png`, `reports/ct_state_topk.csv`, `reports/paper_style_significance.md` |
| MLP vs Attention 路径贡献可分解 | 成立（但与论文方向不同） | `figs/ct_mlp_heatmap.png`, `figs/ct_attn_heatmap.png`, `figs/ct_lineplot_ci.png`, `reports/ct_module_compare.md`, `reports/paper_style_significance.md` |
| modified graph 可区分未来 MLP/Attention 路径作用 | 成立（子集结论） | `figs/modified_graph_mlp_vs_attn.png`, `reports/path_specific_effects.md` |
| 头级 AP/CT 可定位候选关键头 | 成立（子集结论） | `figs/ap_head_heatmap.png`, `figs/ct_head_heatmap.png`, `reports/head_summary.md` |
| 电路具备必要性/充分性/特异性 | 成立（严格全子集） | `figs/final_circuit.png`, `figs/L17H8_probe.png`, `reports/robustness.md` |