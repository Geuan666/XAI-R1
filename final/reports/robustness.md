# Robustness

- 组别：按 prompt 长度三分桶（short/medium/long）。
- 指标：电路必要性（clean drop）、充分性（corrupt restore）、随机头对照 drop。

## Overall (strict full set)
- necessity mean: `0.709888` (95% CI `0.683666`, `0.736159`)
- sufficiency mean: `0.760989` (95% CI `0.738864`, `0.780631`)
- random-head drop mean: `0.012373` (95% CI `0.001791`, `0.022722`)

## Length Buckets
- `short` N=39: necessity=0.660925 [0.613736, 0.700877], sufficiency=0.734509 [0.699554, 0.765141], random_drop=0.009332 [-0.008425, 0.025049]
- `medium` N=38: necessity=0.728637 [0.682201, 0.773593], sufficiency=0.748290 [0.702612, 0.794179], random_drop=0.009201 [-0.004812, 0.026644]
- `long` N=39: necessity=0.740581 [0.701852, 0.780099], sufficiency=0.799843 [0.771807, 0.829313], random_drop=0.018505 [-0.001909, 0.043681]