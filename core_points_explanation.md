# 核心研究点与实验证据对照说明 (Final Submission)

本文档详细说明了实验代码、生成数据及可视化图表如何**层层递进、环环相扣**地支持本研究的三个核心观点。

**数据归档说明**：
*   所有最终版 PDF 图表已整理至：`final_submission/pdfs/`
*   对应的数据源文件 (CSV) 已整理至：`final_submission/data/`

---

## 核心点 1：多源异质证据的结构化解耦 (Structural Disentanglement)

**逻辑链条**：
1.  **提出问题**：异质数据容易导致语义混淆（Feature Entanglement）。此实验旨在证明模型学到的潜变量是**清晰分立**的，而非混沌一团。
2.  **正面证据 (Specialists)**：首先证明模型能学到“专精”的维度。
3.  **反面验证 (Leakage)**：进一步证明这些维度没有“偷看”不该看的信息（即低泄漏），确保了解耦的纯净度。

### 1. 正面证据：专家维度 (Specialist Dimensions)
*   **图表**：`alignment_specialist_dims.pdf`
*   **数据**：`exp1_alignment_specialists.csv`
*   **解读**：
    *   柱状图展示了特定的潜变量维度与特定知识概念之间极高的 Spearman 相关系数值。
    *   **论点**：模型成功分化出了能 1 对 1 响应特定概念的“专家神经元”。

### 2. 反面验证：泄漏检测 (Alignment Leakage) —— *新增解释*
为了证明“专家”不是靠死记硬背（Overfitting）或全局统计（Global Bias），我们需要验证**泄漏 (Leakage)** 情况。
*   **图表**：`alignment_leakage.pdf` & `combo_alignment_leakage.pdf`
    *   **Bubble Plot (左/上)**：展示了每个潜变量维度的“泄漏数”（即该维度与多少个概念强相关）与“最大相关性”的关系。
    *   **ECDF Plot (右/下)**：累积分布函数，量化有多少比例的维度是“低泄漏”的。
*   **数据**：`exp1_alignment_matrix.csv` (用于计算泄漏指标)
*   **深度解读**：
    *   **环环相扣**：如果有高相关性但泄漏也很高（Bubble 都在右侧），说明维度是“万金油”，没有解耦。
    *   **实验结果**：大部分点集中在左侧（Leakage Count 低），且 Bubble 颜色深（Mean Corr 低）但大小不一（Max Corr 高）。
    *   **结论**：这证明了模型实现了**稀疏且精准的解耦**——每个维度只关注它该关注的少数几个概念，而非发生语义漂移。

---

## 核心点 2：稀疏数据下的稳定性 (Stability via Gating & Consistency)

**逻辑链条**：
1.  **提出问题**：教育数据极其稀疏，图结构往往也是有噪的。
2.  **基础保障 (Robustness)**：首先证明**门控融合 (Gating)** 机制在外部噪声冲击下能守住底线。
3.  **进阶优化 (Pareto)**：在稳住底线的基础上，通过**一致性约束 (Consistency)** 进一步压缩表示空间，防止视图漂移，达到最优权衡。

### 1. 基础保障：鲁棒性曲线 (Robustness)
*   **图表**：`combo_robust_pareto.pdf` (左半部分)
*   **数据**：`exp2_robust_curve.csv`
*   **解读**：
    *   随着图结构丢弃率 (Drop Rate, x轴) 从 0% 增加到 40%，AUC/Accuracy (y轴) 保持平稳或仅微幅下降。
    *   **论点**：门控机制有效过滤了噪声边，提供了“结构性抗噪”能力。

### 2. 进阶优化：一致性 Pareto 前沿 (Consistency Trade-off)
*   **图表**：`combo_robust_pareto.pdf` (右半部分)
*   **数据**：`exp2_pareto.csv`
*   **深度解读**：
    *   **环环相扣**：单有抗噪不够，内部表征必须一致。右图展示了 **视图距离 ($D_{view}$)** 与 **误差 ($1-AUC$)** 的博弈。
    *   **实验结果**：随着一致性权重 $\lambda$ 变化，我们观测到一个明显的 Pareto 前沿。
    *   **结论**：我们找到了一个最佳点 $\lambda^*$（图中星号），在此处模型极大压缩了视图差异（消除了不一致性），同时保持了极高的预测精度。这证明了一致性约束是解决稀疏问题的关键补充。

---

## 核心点 3：显式交互建模的可解释性 (Explainability via Interaction)

**逻辑链条**：
1.  **提出问题**：深度学习常被诟病为黑盒，无法解释题目与概念的内在联系。
2.  **微观解释 (Synergy)**：在微观层面，显式画出概念间的协同/拮抗关系。
3.  **宏观验证 (Q-Noise)**：在宏观层面，证明模型不仅仅是画得好看，而是**真正在用**这些结构进行推理。

### 1. 微观解释：协同热力图 (Synergy Heatmap)
*   **图表**：`interaction_heatmap.pdf`
*   **数据**：`exp3_interaction_matrix.csv`
*   **解读**：
    *   热力图中的红色块代表正协同（掌握 A 有助于掌握 B），蓝色代表替代关系。
    *   **论点**：模型不再是黑盒，能输出符合认知科学假设（如先修关系）的显式图谱。

### 2. 宏观验证：Q-矩阵噪声敏感性 (Structure Sensitivity)
*   **图表**：`fig_m3_qnoise_combo.pdf`
*   **数据**：`exp3_qnoise_curve.csv`, `exp3_qnoise_hard_curve.csv`
*   **深度解读**：
    *   **环环相扣**：如果热力图只是“副产品”，那么破坏图结构应该不影响预测。
    *   **实验结果**：当我们通过 Q-Noise (Missing/False/Hard False) 破坏题目-概念映射时，AUC 显著下降（尤其是 Hard False）。
    *   **结论**：这反向证明了模型高度依赖正确的图结构进行推理。模型不仅仅是在做统计拟合，而是在进行**基于图结构的认知推理**。

---

## 附录：数据文件与图表对应索引 (Data-Figure Mapping)

为了便于重绘复现，所有文件均已添加实验编号前缀 (`exp*`)，实现一一对应。

| 实验 | 核心对照点 | PDF 图表文件名 | 对应源数据 CSV | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | 专家维度 | `exp1_alignment_specialist_dims.pdf` | `exp1_alignment_specialists.csv` | 柱状图数据 (维度, 概念, 相关性) |
| **Exp 1** | 泄漏分布 | `exp1_alignment_leakage.pdf`<br>`exp1_combo_alignment_leakage.pdf` | `exp1_alignment_matrix.csv` | 完整的 $R_{dk}$ 矩阵，用于计算 Bubble/ECDF |
| **Exp 2** | 鲁棒性与一致性<br>(组合图) | `exp2_combo_robust_pareto.pdf` | `exp2_robust_curve.csv`<br>`exp2_pareto.csv` | 左图数据: 不同 Drop Rate 下的性能<br>右图数据: 不同 $\lambda$ 下的视图距离与误差 |
| **Exp 3** | 结构依赖性<br>(组合图) | `exp3_qnoise_combo.pdf` | `exp3_qnoise_curve.csv`<br>`exp3_qnoise_hard_curve.csv` | 左图数据: Missing/False 噪声曲线<br>右图数据: Hard False 噪声曲线 |
| **Exp 3** | 协同关系 | `exp3_interaction_heatmap.pdf` | `exp3_interaction_matrix.csv` | 概念间的协同值矩阵 (Synergy Matrix) |

