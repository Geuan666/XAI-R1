# Figure Captions

## Heatmaps
- `figs/ct_state_heatmap.png`：metric=`AIE(state residual restore)`；样本数 `N=116`；x=relative token position，y=layer；RdBu 发散配色，0 居中；|AIE| 99 分位裁剪。
- `figs/ct_mlp_heatmap.png`：metric=`AIE(MLP window restore)`；样本数 `N=116`；窗口=`[l-4..l+5]`；x=relative token position，y=layer；RdBu 发散配色，0 居中；|AIE| 99 分位裁剪。
- `figs/ct_attn_heatmap.png`：metric=`AIE(Attention window restore)`；样本数 `N=116`；窗口=`[l-4..l+5]`；x=relative token position，y=layer；RdBu 发散配色，0 居中；|AIE| 99 分位裁剪。
- `figs/ap_head_heatmap.png`：metric=`p_clean - p_clean[head<-corrupt]`；样本数 `N=16`；x=head，y=layer；RdBu 发散配色，0 居中；|IE| 99 分位裁剪。
- `figs/ct_head_heatmap.png`：metric=`p_corrupt[head<-clean] - p_corrupt`；样本数 `N=16`；x=head，y=layer；RdBu 发散配色，0 居中；|IE| 99 分位裁剪。

## Line / Composite
- `figs/avg_trace_triptych.png`：三联图（State/MLP/Attention）；样本数 `N=116`；均值 AIE 热图。
- `figs/ct_lineplot_ci.png`：layer-profile 线图；样本数 `N=116`；阴影=bootstrap 95% CI。
- `figs/modified_graph_mlp_vs_attn.png`：line+band；样本数 `N=24`；实线=均值，阴影=样本 95% 分位区间。

## Other Figures
- `figs/L17H8_probe.png`：metric=`P(<tool_call> at t=1)`；样本数 `N=116`；柱高=均值，误差条=bootstrap 95% CI。
- `figs/final_circuit.png`：节点为候选电路头 `L17H8, L20H5, L21H1, L21H12`；边宽按组合分数缩放，用于结构可视化。