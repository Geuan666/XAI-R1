# Head-level AP/CT Summary

- 头级扫描样本：严格子集（stratified），`N=20`。
- `ap_head_heatmap.png` 与 `ct_head_heatmap.png` 使用红蓝发散色，0 居中，99 分位裁剪。
- AP 定义：`p_clean - p_clean[head<-corrupt]`；CT 定义：`p_corrupt[head<-clean] - p_corrupt`。

## Top Heads
- `L21H1`: score=0.606887, AP=0.248420, CT=0.358467, bootstrap_topk_freq=1.000
- `L21H12`: score=0.590790, AP=0.236730, CT=0.354060, bootstrap_topk_freq=1.000
- `L17H8`: score=0.535273, AP=0.284117, CT=0.251156, bootstrap_topk_freq=1.000
- `L24H6`: score=0.436428, AP=0.243606, CT=0.192822, bootstrap_topk_freq=1.000
- `L20H5`: score=0.303985, AP=0.153220, CT=0.150765, bootstrap_topk_freq=0.000
- `L23H6`: score=0.244685, AP=0.087271, CT=0.157414, bootstrap_topk_freq=0.000
- `L17H2`: score=0.217239, AP=0.113519, CT=0.103720, bootstrap_topk_freq=0.000
- `L16H13`: score=0.202679, AP=0.129121, CT=0.073557, bootstrap_topk_freq=0.000
- `L19H8`: score=0.182814, AP=0.093823, CT=0.088991, bootstrap_topk_freq=0.000
- `L20H14`: score=0.180883, AP=0.092771, CT=0.088111, bootstrap_topk_freq=0.000
- `L18H15`: score=0.129176, AP=0.077581, CT=0.051595, bootstrap_topk_freq=0.000
- `L19H6`: score=0.126656, AP=0.047267, CT=0.079389, bootstrap_topk_freq=0.000
- `L18H7`: score=0.109261, AP=0.053233, CT=0.056028, bootstrap_topk_freq=0.000
- `L15H9`: score=0.101451, AP=0.061676, CT=0.039776, bootstrap_topk_freq=0.000
- `L17H11`: score=0.095868, AP=0.052321, CT=0.043547, bootstrap_topk_freq=0.000
- `L19H5`: score=0.094395, AP=0.050028, CT=0.044367, bootstrap_topk_freq=0.000
- `L16H7`: score=0.092740, AP=0.054893, CT=0.037846, bootstrap_topk_freq=0.000
- `L16H5`: score=0.088517, AP=0.063565, CT=0.024952, bootstrap_topk_freq=0.000
- `L15H5`: score=0.086365, AP=0.063432, CT=0.022933, bootstrap_topk_freq=0.000
- `L16H4`: score=0.079680, AP=0.057532, CT=0.022148, bootstrap_topk_freq=0.000

## Selected Circuit Heads
- `L17H8, L21H1, L21H12, L24H6`