## Project Goal
- 目标：把论文中的 Causal Tracing/组件定位/电路验证流程迁移到本项目，解释模型为何在第一个生成 token 选择 `<tool_call>`。
- 研究对象：`clean prompt` 与 `corrupt prompt` 的配对样本（同一 `q`）。
- 最终交付：可复现的图、表、结论，尤其是 `ap_head_heatmap.png`、`ct_head_heatmap.png`、`final_circuit.png`。

## Key Folders
- `pair/`：主数据集（`prompt-clean-q*.txt`、`prompt-corrupted-q*.txt`、`meta-q*.json`、已有首 token 评估 CSV）。
- `sample/`：已采样到的显著单样本点；进行单样本研究（case study）时优先使用该目录，也可用于冒烟测试。
- `src/`：实验脚本与分析代码。
- `figs/`：最终图像输出目录。
- `reports/`：指标表、结论文档、实验记录。
- `Interpretability in the Wild/`：参考论文源码与图注（实验设计对照依据）。

## Non-Negotiable Rules
- 第一生成 token 是唯一决策位点：主指标必须围绕 `P(<tool_call> at t=1)`。
- 命名规范必须遵守（见 `todo.md`）：
  - 头：`L{layer}H{head}`
  - MLP：`MLP{layer}`
  - 强制图名：`ap_head_heatmap.png`、`ct_head_heatmap.png`、`final_circuit.png`
- 热力图必须用红蓝发散配色，0 居中，正负贡献可直接分辨。
- 图注必须写清指标定义、样本数、归一化/裁剪方式、置信区间设置。
- CT 主图组织对齐论文：`Layer x Relative Position`（不以 `Sample x Layer` 作为主展示图）。
- 模块对比与路径对比除 bootstrap CI 外，需补充置换/符号翻转 `p-value`。
- 若任务是单样本研究，默认使用 `sample/`；只有在报告中写明理由时，才可改用 `pair/` 的其他单例样本。

## Execution Guidance
- 先做最小闭环：`sample/` 冒烟 -> `pair/` 全量。
- 子集实验默认采用长度分层抽样（stratified），避免短样本偏置。
- 所有关键结论必须包含：`N`、均值、95% bootstrap CI。

## Environment & Runtime Constraints
- Python 环境：`base`。
- 计算资源：优先使用 `GPU 4090 24G`。
- 若显存不足：
  - 首选等待资源释放并重试；
  - 不要默认切 CPU 长跑；
  - 若必须降配，先在 `reports/` 记录原因与影响。
- 模型：`/root/data/Qwen/Qwen3-1.7B`
- 图像风格最好与参考论文一致
- 只在你的工作目录下读写，不要关注其他代码
