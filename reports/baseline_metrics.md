# Baseline Metrics

- 主指标：`p_tool = P(<tool_call> at t=1)`；二值标签 `y = 1[top1 == <tool_call>]`。
- 总样本数（对齐后）：`N=164`；严格子集：`N=116`。

## Full Set (All Aligned Pairs)
- clean top1=`<tool_call>`: `144/164 = 87.805%`
- corrupt top1!=`<tool_call>`: `136/164 = 82.927%`
- strict pair success: `116/164 = 70.732%`
- `TE = p_clean - p_corrupt` mean: `0.60536` (95% CI `0.56374`, `0.64435`)

## Strict Subset
- `TE` mean: `0.74093` (95% CI `0.71241`, `0.76774`)

## Sanity Check vs Provided CSV
- provided clean=0.896, corrupt_non_tool=0.835, strict=0.732; recomputed clean=0.878, corrupt_non_tool=0.829, strict=0.707

## Figure/Metric Notes
- 全部指标只用第一个生成位置 `t=1`。
- 置信区间为 bootstrap 95% CI。
- `te_summary.csv` 已保存每个样本的 `p_clean / p_corrupt / TE / strict`。