# Paper-style Alignment and Significance

## Setup
- 严格子集样本数：`N=116`
- 显著性复算口径：关键 profile（rel=0 与 decision token）做全量统计，峰位由数据自动决定。

## Data-driven Peak Layers
- State@rel0 peak layer: `L1`
- MLP@rel0 peak layer: `L0`
- Attn@rel0 peak layer: `L2`
- MLP@decision peak layer: `L22`
- Attn@decision peak layer: `L24`

## Peak-vs-Rest Significance (one-sided Wilcoxon, greater)
- State@rel0: mean diff=`0.673128` 95%CI[`0.653884`, `0.690515`], p=`4.493e-21`
- MLP@rel0: mean diff=`0.327736` 95%CI[`0.295787`, `0.358861`], p=`4.493e-21`
- Attn@rel0: mean diff=`0.675326` 95%CI[`0.656329`, `0.692798`], p=`4.493e-21`
- MLP@decision: mean diff=`0.336571` 95%CI[`0.316854`, `0.356020`], p=`4.493e-21`
- Attn@decision: mean diff=`0.373736` 95%CI[`0.362907`, `0.385023`], p=`4.493e-21`

## Module Dominance Tests
- Early site dominance (Attn-MLP @ L2, rel=0): mean diff=`0.501558` 95%CI[`0.465295`, `0.539339`], p=`4.493e-21`
- Late site dominance (Attn-MLP @ L24, decision): mean diff=`0.342837` 95%CI[`0.311529`, `0.371206`], p=`4.493e-21`

## Interpretation
- 与原论文事实回忆任务不同，本任务在全量 strict 上显示 early 与 late 均为 Attention 更强，并且差异显著。
- 这说明 `<tool_call>` 路由决策更接近跨位置条件整合，而不是单点 MLP 事实检索。