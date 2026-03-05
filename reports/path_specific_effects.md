# Path-specific Effects (Modified Graph)

- 样本：严格子集分析集（分层抽样），`N=28`。
- `full_restore`：仅做 residual restore。
- `freeze_future_mlp`：restore 同时把未来层 MLP 输出固定为 corrupted baseline。
- `freeze_future_attn`：restore 同时把未来层 Attention 输出固定为 corrupted baseline。

## Summary
- full_restore mean IE: `0.184798` (95% CI `0.165999`, `0.204820`), p=`3.332e-04`
- freeze_future_mlp mean IE: `0.056449` (95% CI `0.049444`, `0.063574`), p=`3.332e-04`
- freeze_future_attn mean IE: `0.058505` (95% CI `0.050991`, `0.066069`), p=`3.332e-04`
- paired test: full vs freeze_future_mlp p=`3.332e-04`
- paired test: full vs freeze_future_attn p=`3.332e-04`

## Figure Caption
- `figs/modified_graph_mlp_vs_attn.png`: 横轴层号，纵轴 IE；实线为均值，阴影为样本分布 95% 分位区间。