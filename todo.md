# Tool-Call 决策可解释性复现实验 TODO（仅定位与寻找）

> 参考论文：`Interpretability in the Wild` 目录下的 LaTeX（只迁移“信息流定位/因果追踪”部分）。  
> 本文档目标：把论文中的定位实验迁移到当前项目（`clean -> <tool_call>`、`corrupt -> 非 <tool_call>`）。
> 模型：`/root/data/Qwen/Qwen3-1.7B`
## Scope（强制）
- 只做定位与寻找：Causal Tracing、Activation Patching、组件排名、候选电路识别与验证。
- 不做知识编辑：不做权重改写、不做 ROME/FT/MEND/KE 等编辑效果对比。
- 不做应用评测：不做生成质量、人评、下游任务应用评测。

## Part A. 需要完成的目标（必须交付）

### A0. 任务定义与验收口径（先统一）
- 研究问题：模型为什么在第一个生成 token 选择 `<tool_call>`。
- 统一标签：
  - `y=1`：第一个生成 token 的 `top1 == <tool_call>`
  - `y=0`：第一个生成 token 的 `top1 != <tool_call>`
- 基本样本：`pair/prompt-clean-q*.txt` 与 `pair/prompt-corrupted-q*.txt` 一一配对。
- 主指标全部围绕第一个生成位置 `t=1`，不混入后续 token。
- 当前数据基线（来自 `pair/first_token_len_eval_qwen3_1.7b.csv`，共 164 对）：
  - clean 中 `top1=<tool_call>`：`147/164 = 89.6%`
  - corrupt 中 `top1!=<tool_call>`：`137/164 = 83.5%`
  - 严格成对成功（clean 是 `<tool_call>` 且 corrupt 非 `<tool_call>`）：`120/164 = 73.2%`
- 验收口径：
  - 主分析默认报告“全量 164 + 严格子集 120”两套结果；
  - 任何只在筛后子集成立的结论，都必须标注“子集结论”。

### A1. 复现“Clean / Corrupted / Restore”主实验框架（论文核心）
- 对每个 pair 执行三次运行：
  - Clean run：输入 clean prompt，记录全层激活与 `P(<tool_call>)`。
  - Corrupted run：输入 corrupt prompt，记录全层激活与 `P(<tool_call>)`。
  - Corrupted-with-restoration run：在 corrupt 前向中，将某个中间状态替换为 clean 对应状态，观测 `P(<tool_call>)` 恢复量。
- 产物：
  - `reports/baseline_metrics.md`：clean/corrupt 成功率、分布、失败样本统计。
  - `reports/te_summary.csv`：每个样本的 `TE`（见指标规则）。

### A2. 复现“因果追踪热图”（state-level）
- 对 `(token position i, layer l)` 做 restoration 扫描，得到状态级因果图（AIE/IE 热力）。
- 目标结论（迁移版）：
  - 找到与 `<tool_call>` 决策最相关的“早期位点”和“晚期位点”；
  - 判断关键证据是更偏 MLP 还是更偏注意力路径。
- 产物：
  - `figs/ct_state_heatmap.png`（建议保留，便于与 head 图对照）
  - `reports/ct_state_topk.csv`

### A3. 复现“MLP vs Attention 分解”
- 分别恢复 MLP 输出与注意力输出，比较其对 `P(<tool_call>)` 的贡献。
- 按论文做法支持窗口恢复（默认 10 层窗口，可调整并记录）。
- 产物：
  - `figs/ct_mlp_heatmap.png`
  - `figs/ct_attn_heatmap.png`
  - `reports/ct_module_compare.md`

### A4. 复现“修改计算图干预”（path-specific）
- 做两组图：
  - 冻结/切断未来 MLP 路径，再测 restoration 效果；
  - 冻结/切断未来 Attention 路径，再测 restoration 效果。
- 目标结论：判定 tool-call 决策是否依赖中层 MLP 计算链（对应论文 `modified graph intervention` 思路）。
- 产物：
  - `figs/modified_graph_mlp_vs_attn.png`
  - `reports/path_specific_effects.md`

### A5. 头级定位：AP 与 CT 两张核心图（强制文件名）
- 头级 Activation Patching（AP）与 Causal Tracing（CT）都要做。
- 强制输出：
  - `figs/ap_head_heatmap.png`
  - `figs/ct_head_heatmap.png`
- 目标：锁定候选关键头（后续 probe 与电路构建输入）。

### A6. 组件探针与电路构建
- 对 top 组件做 probe（至少覆盖 top attention heads，必要时含 MLP）。
- 强制命名示例：
  - `figs/L7H14_probe.png`
- 构建最终可解释电路图（节点+边）：
  - 强制输出 `figs/final_circuit.png`
- 必须做三类验证：
  - 必要性：只在 clean 中打掉该电路，`P(<tool_call>)` 显著下降；
  - 充分性：只保留/恢复该电路，corrupt 中 `P(<tool_call>)` 明显回升；
  - 特异性：打掉非电路组件影响应明显更小。

### A7. 鲁棒性与负对照（至少一组）
- 至少做 1 组鲁棒性：
  - 不同 corruption 强度/方式；
  - 或不同 prompt 长度桶；
  - 或随机组件对照（random heads / random layers）。
- 产物：
  - `reports/robustness.md`

### A8. 结论对齐表（仅对齐定位章节）
- 输出 `reports/conclusion_alignment.md`，按“论文定位结论 -> 本项目是否成立 -> 证据图/表”逐项对齐。
- 仅覆盖：clean/corrupt/restore、IE/AIE、MLP-vs-Attn、path-specific、头级与电路定位。

---

## Part B. 规则（目标与规则必须明确，但方法可发挥）

### R1. 命名规范（强制）
- 注意力头：`L{layer}H{head}`，例：`L7H14`
- MLP：`MLP{layer}`，例：`MLP12`
- 其他结点（如 residual）：`RESID_L{layer}`（如确实需要，电路中不出现）
- 文件命名：
  - 热力图：`ap_head_heatmap.png` / `ct_head_heatmap.png`
  - 组件探针图：`L7H14_probe.png`
  - 最终电路图：`final_circuit.png`

### R2. 颜色与对比（强制）
- 所有热力图必须使用红/蓝对比的发散色系（如 `RdBu` 或同类）。
- 强制要求：
  - 0 值在中间（白或浅色）；
  - 正负贡献一眼可区分；
  - 对比度足够（可做裁剪/标准化，但必须写进图注）。

### R3. 图排版与风格（强制）
- 风格必须“清爽、可读、留白足够”。
- 字体统一，字号层级明确：标题 > 轴标签 > tick。
- 轴标签必须清楚写出维度（如 `Layer`、`Token Position`、`Head`）。
- 图注必须写清：指标定义、归一化方式、样本数、是否含置信区间。
- 论文对齐要求：
  - `ct_state_heatmap / ct_mlp_heatmap / ct_attn_heatmap` 应统一为 `Layer x Relative Position` 组织，不使用 `Sample x Layer` 作为主图。
  - 需提供至少一张 line plot（含 95% CI）用于对照论文附录线图风格。

### R4. 指标规则（强制）
- 主指标：`p_tool = P(<tool_call> at t=1)`。
- 二值决策：`y = 1[p_tool is top1]`。
- 总效应：`TE = p_tool(clean) - p_tool(corrupt)`。
- 间接效应：`IE(m) = p_tool(corrupt + restore m_from_clean) - p_tool(corrupt)`。
- 平均间接效应：`AIE = mean_q IE_q`（按样本平均）。
- 建议同时报 `NIE = IE / (TE + eps)`，便于跨样本比较（可选但推荐）。

### R5. 统计规则（强制）
- 所有核心结论必须给：
  - 样本数 `N`
  - 均值
  - 95% CI（bootstrap）
- 核心差异比较（如 MLP vs Attn、full vs freeze）需补充置换/符号翻转检验 `p-value`。
- 关键组件入选不能只看单次峰值，至少要满足“多次种子/噪声设置下稳定”。

### R6. 数据与切分规则（强制）
- 优先使用“有效样本子集”：`clean top1=<tool_call>` 且 `corrupt top1!=<tool_call>`。
- 保留完整样本统计作为附录，不得只报筛后结果。
- clean/corrupt 必须严格按 `q` 配对，不允许打乱对齐。
- 分析子集抽样默认使用长度分层抽样（stratified），避免只取短样本造成分布偏差。
- 单样本研究规则：若做 case study / 单样本定位，默认使用 `sample/` 目录中的样本（该目录为已筛到的显著点）；除非有明确理由，不使用 `pair/` 中随机单例替代。

### R7. 结果落盘规则（强制）
- 图统一落 `figs/`，表与文字结论落 `reports/`。
- 每张图需可追溯到生成脚本与配置（脚本里保存 seed、模型版本、数据版本）。
- 不覆盖同名历史结果时，需在 `reports/` 记录本次版本号或时间戳。

### R8. 可发挥空间（鼓励）
- 允许自由扩展的方法：
  - Path patching、activation steering、head clustering、稀疏回归探针等。
- 但扩展必须满足两条底线：
  - 不改变 R4 的主指标定义；
  - 不替代 A1~A8 的主线交付，只能作为增强证据。

---

## Part C. 建议执行顺序（可并行）

- [ ] Phase 1：跑通 A0/A1（基线与 TE）
- [ ] Phase 2：完成 A2（state-level causal trace）
- [ ] Phase 3：完成 A3/A4（模块分解 + modified graph）
- [ ] Phase 4：完成 A5/A6（head 图 + probe + final circuit）
- [ ] Phase 5：完成 A7/A8（鲁棒性 + 结论对齐）

> 判定“完成”的标准不是代码跑过，而是图、表、结论三者一致，并能回答：  
> “哪些内部组件在驱动 `<tool_call>` 首 token 决策，它们是必要/充分的吗？”
