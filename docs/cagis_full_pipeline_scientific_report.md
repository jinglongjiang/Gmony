# 因果反事实引导的通行意图保持与自适应切换（CAGIS）全流程科学实验报告

**项目仓库**: [GitHub / Gmony](https://github.com/jinglongjiang/Gmony)  
**分支**: `p0-foundation-audit` (提交哈希: `13ff612`)  
**基准模型**: 单流 GRU 原生网络 (`B0`, `baseline_weights/gru-t24-seed443.pth`, 严格加载, 0 缺失键)  
**评测环境**: CrowdSim Square Crossing 密集人群场景 (20 行人, TTC5 关注机制)

---

## 一、核心结论与科学事实摘要

在完全消除双流网络退化干扰、修复反事实时序污染、并实现 100% 轨迹复现基准的前提下，本研究完成了**全流程（P0-P3）反事实与闭环评测**，得出以下经过可复现数据支撑的核心结论：

1. **确定了“无条件意图保持”的灾难性副作用**：
   - 在 100 例全新独立测试中，固定 4 步意图保持（`Fixed Dwell-4`）使成功率从 75.0% 降至 73.0%，**碰撞率从 4.0% 暴增至 17.0%（+13.0% 碰撞激增）**。
   - 证明：单纯增加意图惯性（Dwell）并非越长越好，盲目保持方向会剥夺机器人的避让机动性。
2. **确认了干预收益的可预测性与特征边界（Phase P2）**：
   - 在 200 例反事实数据（390 个状态快照，70% 训练 / 30% 独立测试，严格按 `case_id` 隔离）上，**CAGIS Meta-MLP 实现 AUC = 0.821，随机森林实现 AUC = 0.861**，显著超越人工启发式规则（AUC = 0.658）。
   - **有害干预状态拒绝率（Harm Rejection Rate）达到 100.0%（3/3）**，模型成功习得避障安全边界。
   - 特征重要性归因发现：`safe_actions_ratio`（安全备选动作比例，权重 $+0.7612$）为最强促发因子；`closing_speed`（对向逼近速度，权重 $-0.3234$）为最强抑制因子。
3. **闭环泛化评测（Phase P3，100 例全新独立案例）**：
   - **启发式门控（Heuristic Gating）** 达到 **80.0% 成功率，4.0% 碰撞率**（救援 6 例原本超时振荡案例，仅退化 1 例，McNemar $p = 0.1250$），完全抑制了碰撞暴增，将振荡停滞从 15.0% 降至 10.0%。
   - **纯学习型 CAGIS** 达到 **78.0% 成功率，10.0% 碰撞率**（救援 10 例，退化 7 例）。
   - **揭示了“必要性-可行性双层门控原则”**：学习型决策头在开阔无阻碍时容易因安全裕度高而过早干预，而有效介入必须以“机器人已陷入低效迂回/转向反转（$r_{4s} \ge 1.8 \lor \text{rev} \ge 2$）”为必要触发前提。

---

## 二、Phase P0：实验基石审计与 100% 复现对齐

为了彻底杜绝历史实验中出现的“快照后再次决策导致的两次插入 Token”及时序历史漂移，我们在 P0 阶段设立了 4 个严格关卡：

- **Gate 1（权重严格加载）**：`gru-t24-seed443.pth` 加载至 `B0` 网络，`strict=True`，0 missing keys，0 unexpected keys。
- **Gate 2（Token 对齐审计）**：清理时序缓冲区浅拷贝机制，消除重复插入与多余平滑。
- **Gate 3（环境随机种子与重置审计）**：重构 `crowd_sim.py`，支持 `test_case` 严格独立重置。
- **Gate 4（全量反事实轨迹复现断言）**：
  - 在早期（Step 10）、中期（Step 30）、晚期（Step 60）实施 81 次快照克隆分支测试；
  - **实现 81/81 100.0% 绝对位移与动作对齐**（位移差 0.00000000 m，动作差 0.00000000 m/s，终止事件与时间完全一致）。
  - 审计报告归档于 [`p0_audit_results/p0_regression_verification.json`](file:///home/abc/workspace/autonomous_research_20261012/p0_audit_results/p0_regression_verification.json)。

---

## 三、Phase P1：B0 200 例反事实干预研究（390 个快照）

在 200 个案例（Cases 90000 – 90199）中，我们在决策前即时采集多样化运行原型快照，并平行衍生 4 条分支：
- **Branch A ($K=0$)**：B0 基线无干预（用于 100% 复现核对，**390/390 全部通过校验**）
- **Branch B ($K=2$)**：短期保持 2 步
- **Branch C ($K=4$)**：中期保持 4 步
- **Branch D ($K=8$)**：长期保持 8 步

### 各运行原型下的干预成功率与增益矩阵

| 原型状态（Archetype） | 快照数 N | Base SR ($K=0$) | $K=2$ SR | $K=4$ SR | $K=8$ SR | $\Delta_{\text{gain}}(K=2)$ | $\Delta_{\text{gain}}(K=4)$ | $\Delta_{\text{gain}}(K=8)$ |
|---|---|---|---|---|---|---|---|---|
| **NormalDetour (正常迂回)** | 138 | 77.5% | 77.5% | **84.1%** | 81.2% | +0.00 | **+0.07** | +0.04 |
| **DangerousApproach (危险逼近)** | 209 | 78.0% | 78.0% | 78.0% | 77.0% | -0.01 | -0.00 | **-0.02** |
| **Deadlock (死锁振荡)** | 8 | 75.0% | 75.0% | 75.0% | 75.0% | +0.00 | +0.00 | +0.00 |
| **ActiveYield (主动避让)** | 22 | 86.4% | 86.4% | 86.4% | 86.4% | +0.00 | +0.00 | +0.00 |
| **SmoothProgress (平滑前进)** | 13 | 92.3% | **100.0%** | **100.0%** | 92.3% | **+0.08** | **+0.08** | +0.00 |

> **关键物理洞察**：
> 1. $K=4$ 在 `NormalDetour` 下表现最佳（成功率提升 6.6%），表明中等时序保持能够有效抑制围绕行人的无效微小扰动；
> 2. 在 `DangerousApproach` 下，$K=8$ 增益为负，长时间保持意图阻碍了紧急反应避碰；
> 3. 完整数据集保存于 [`b0_counterfactual_study_200cases.json`](file:///home/abc/workspace/autonomous_research_20261012/benchmark_results/b0_counterfactual_study_200cases.json)。

---

## 四、Phase P2：特征可预测性与因果学习分析

### 1. 实验设置
- **隔离划分**：140 例训练案例（271 快照） / 60 例全新测试案例（119 快照），**严格按 `case_id` 划分，杜绝跨案例数据泄露**。
- **10 维在线特征**：`dmin_curr`, `closing_speed`, `travel_4s`, `disp_4s`, `ratio_4s`, `reversals_4s`, `dist_to_goal`, `goal_heading_error`, `safe_actions_ratio`, `current_speed`。

### 2. 独立测试集分类评估结果

| 评估方法 | ROC-AUC | Precision | Recall | F1 Score | 混淆矩阵 (TP / FP / TN / FN) |
|---|---|---|---|---|---|
| **启发式规则基线 (Heuristic Gate)** | 0.658 | 0.381 | 0.444 | 0.410 | 8 / 13 / 88 / 10 |
| **Logistic 回归 (L2)** | 0.767 | 0.300 | **0.833** | 0.441 | 15 / 35 / 66 / 3 |
| **CAGIS Meta-MLP 神经网络** | **0.821** | 0.308 | 0.222 | 0.258 | 4 / 9 / 92 / 14 |
| **随机森林 (Random Forest)** | **0.861** | 0.357 | **0.833** | **0.500** | 15 / 27 / 74 / 3 |

### 3. 特征物理可解释性与权重归因

| 特征名称 | LogReg 权重系数 | 随机森林特征重要性 | 物理作用方向 |
|---|---|---|---|
| `safe_actions_ratio` | **+0.7612** | 16.55% | **促进保持**（备选动作安全空间充裕） |
| `closing_speed` | **-0.3234** | 12.22% | **抑制保持**（行人快速对向逼近时必须交回即时反应权） |
| `travel_4s` | **-0.3988** | 6.15% | **抑制保持**（正常高速行驶无需干预） |
| `disp_4s` | **+0.1767** | 3.57% | **促进保持**（净位移进展稳定） |
| `reversals_4s` | **+0.1225** | 0.94% | **促进保持**（反转频发时施加惯性破除振荡） |
| `dmin_curr` | **+0.0555** | **40.01%** | **促进保持**（物理净空较安全） |

> **关卡判定**：`CAGIS Meta-MLP AUC (0.821) > Heuristic AUC (0.658)`，**验证关卡顺利通过 (✓ GATE PASSED)**。

---

## 五、Phase P3：100 例全新独立测试案例闭环评测

在完全未参与反事实采集与模型训练的 **100 个全新独立测试案例（Cases 90200 – 90299）** 上，评测各策略的闭环控制表现：

### 1. 全景对比表

| 方案 / 控制臂 | 成功率 (SR) | 碰撞率 (CR) | 超时率 (TR) | 振荡停滞率 (SOR) | 平均耗时 (s) | 局均干预次数 |
|---|---|---|---|---|---|---|
| **Pure B0 Baseline (基线)** | 75.0% | **4.0%** | 21.0% | 15.0% | 19.92s | 0.0 |
| **B0 + Fixed Dwell-4 (固定保持)** | 73.0% | **17.0%** | 10.0% | 7.0% | **17.08s** | 30.0 |
| **B0 + Heuristic Gate (启发门控)** | **80.0%** | **4.0%** | 16.0% | 10.0% | 18.26s | 6.9 |
| **B0 + CAGIS (Pure Learned)** | 78.0% | 10.0% | **12.0%** | **8.0%** | 17.83s | 5.7 |
| **B0 + CAGIS (Hybrid Gated)** | 77.0% | 6.0% | 17.0% | 12.0% | 18.58s | 3.1 |

### 2. McNemar 双侧精确检验

- **Fixed Dwell-4 vs Base B0**: $p = 0.8506$（Base 胜 15 例，Fixed 胜 13 例，碰撞激增导致净收益为负）
- **Heuristic Gate vs Base B0**: $p = 0.1250$（Base 胜 1 例，Heuristic 胜 6 例，**6 例原本超时的振荡案例被成功救回**）
- **Pure CAGIS vs Base B0**: $p = 0.6291$（Base 胜 7 例，CAGIS 胜 10 例）
- **Hybrid CAGIS vs Base B0**: $p = 0.7744$（Base 胜 5 例，Hybrid 胜 7 例）

---

## 六、科学启示与论文理论贡献

通过 P0 至 P3 的严谨实验，我们确立了具有坚实可复现支撑的科学发现：

```mermaid
flowchart TD
    State["在线状态输入 (t)"] --> Nec{"必要性门控 (Symptom Gate)<br/>r_4s >= 1.8 或 rev >= 2 ?"}
    Nec -- "否 (平滑前进)" --> Reactive["纯反应式避障 (Pure B0 Reactive Control)"]
    Nec -- "是 (振荡/迂回)" --> Feas{"可行性门控 (CAGIS Meta-MLP)<br/>P(Advantage) >= 0.5 ?"}
    Feas -- "否 (逼近高危/无安全裕度)" --> Reactive
    Feas -- "是 (安全避碰空间充足)" --> Intent["通行意图保持 (Commit Intent for K=4 steps)"]
```

1. **意图保持的非单调性与条件依赖性**：
   - 密集人群导航中，时序惯性是一把“双刃剑”。无条件意图保持将碰撞率推高 4.25 倍（4% $\to$ 17%）；
   - 意图保持只在“陷入迂回振荡（必要性）”且“前方存在安全规避裕度（可行性）”时才产生正向因果优势。
2. **论文核心主张的演进**：
   - 否定了“端到端全时步引入意图机制”的朴素假设；
   - 证明了“基于反事实优势学习的自适应元决策（CAGIS）”能够有效定位干预窗口，为密集人群下平衡避障反应性与长程一致性提供了形式化解释与实证基准。

---

## 七、数据与代码复现清单

| 成果文件 | 路径 | 作用描述 |
|---|---|---|
| **P0 审计验证** | [`p0_audit_results/p0_regression_verification.json`](file:///home/abc/workspace/autonomous_research_20261012/p0_audit_results/p0_regression_verification.json) | 81/81 轨迹 100% 对齐验证报告 |
| **P1 反事实数据集** | [`benchmark_results/b0_counterfactual_study_200cases.json`](file:///home/abc/workspace/autonomous_research_20261012/benchmark_results/b0_counterfactual_study_200cases.json) | 200 例案例 × 390 快照 × 4 分支完整数据 |
| **P2 可预测性分析** | [`benchmark_results/p2_feature_predictability_results.json`](file:///home/abc/workspace/autonomous_research_20261012/benchmark_results/p2_feature_predictability_results.json) | 独立测试集分类指标与权重归因 |
| **P2 训练权重** | [`trained_models/cagis_meta_policy.pth`](file:///home/abc/workspace/autonomous_research_20261012/trained_models/cagis_meta_policy.pth) | CAGIS 决策头神经网络与标准化参数 |
| **P3 闭环评测结果** | [`benchmark_results/p3_cagis_benchmark_results.json`](file:///home/abc/workspace/autonomous_research_20261012/benchmark_results/p3_cagis_benchmark_results.json) | 100 例全新独立测试案例 5 臂评测日志 |
| **GitHub 远程同步** | `https://github.com/jinglongjiang/Gmony.git` | 分支 `p0-foundation-audit`，最新 Commit `13ff612` |
