# 预注册科学研究方案与判定判据 (PREREG.md)

本文件在执行任何测试集评估前提交并冻结，严格约束实验执行范围、对比判据、统计检验方法与阶段流转规则。

---

## 一、阶段 1：学习型意图承诺门控复现方案

### 1. 核心比较对象
- **基线模型（Arm 1: Base B0）**：单流 GRU 冻结底座（`baseline_weights/gru-t24-seed443.pth`），无任何干预。
- **主比较对象（Arm 4: Learned LR Gated）**：逻辑回归门控（K=4），基于冻结模型与阈值（`trained_models/champion_gating_lr.joblib`，阈值 0.40）。
- **汇报对比组（不参与复现判据）**：
  - Arm 2: Heuristic Gated（人工门控，K=4）
  - Arm 3: Learned RF Gated（随机森林门控，K=4）

### 2. 测试数据集与环境配置
- **主测试块**：Case 97000 – 97999（共 1000 例，Square Crossing 方形交叉场景，20 行人，时间上限 50.0s，步长 0.25s）。
- **泛化观察块**：Case 97000 – 97499（共 500 例，Circle Crossing 圆形交叉场景，20 行人），仅做数据汇报，不参与判定。
- **严格隔离区间**：严禁读取或运行 Case 92000–92499、90000–90299、93000–93199、95000–96499。

### 3. 回归与冒烟前置闸门
1. **回归闸门（Regression Gate）**：
   在 Case 90300 – 90309（10 例）上重跑四臂，逐回合终局结局（`outcome`）、耗时（`final_time`）必须与 `benchmark_results/stage4_500cases_final_evaluation.json` **100% 一致**。任何不一致立即终止任务并排查，不进行后续测试。
2. **冒烟闸门（Smoke Gate）**：
   在主测试块 Case 97000 – 97009（10 例）上先行冒烟，确认多进程执行、数据落盘与事件统计均正常，方可启动全量 1000 例测试。

### 4. 主假设判定判据（仅对主测试块 1000 例中的 Arm 4 对 Arm 1）
- **指标定义**：
  - 成功率差：$\Delta\text{SR} = \text{SR}_{\text{LR}} - \text{SR}_{\text{Base}}$
  - 碰撞率差：$\Delta\text{CR} = \text{CR}_{\text{LR}} - \text{CR}_{\text{Base}}$
  - 配对精确检验：双侧 McNemar 精确二项检验，统计非一致对 $(b, c)$，计算 $p$ 值：
    $$p = 2 \times \sum_{k=0}^{\min(b,c)} \binom{b+c}{k} 0.5^{b+c}$$
  - 碰撞率差 95% 置信区间：基于配对样本差异的精确/Score 置信区间 $[\text{CI}_{\text{low}}, \text{CI}_{\text{up}}]$。
- **状态词判定规则**：
  - **`REPLICATED`**：同时满足：
    1. 成功率差大于 0：$\Delta\text{SR} > 0$；
    2. 配对检验统计显著：双侧 $p < 0.05$；
    3. 碰撞率增量受控：$\Delta\text{CR}$ 的 95% 置信区间上界 $\text{CI}_{\text{up}} < +2.0$ 个百分点。
  - **`NOT REPLICATED`**：
    - 成功率差点估计小于 +1.0 个百分点：$\Delta\text{SR} < +1.0\%$。
  - **`INCONCLUSIVE`**：
    - 其余所有情况（例如：$\Delta\text{SR} \ge +1.0\%$，但 $p \ge 0.05$，或碰撞率区间上界超标）。

### 5. 流转规则
- 若判定为 **`REPLICATED`**：编写阶段 1 报告，进入阶段 2（可学习终止）。
- 若判定为 **`NOT REPLICATED`** 或 **`INCONCLUSIVE`**：编写阶段 1 报告，**停止实验，不做阶段 2**。

---

## 二、阶段 2：可学习终止方案（仅在阶段 1 判定为 REPLICATED 时执行）

### 1. 结构契约
- 保持 `Stage4ControllerPolicy` 整体架构不变（母体价值前瞻原样保留，通过对候选打分加偏置起作用，门控触发与意图推导不变）。
- 将“固定承诺 4 步”换为轻量级终止头（Termination Head），在每个承诺步实时评估当前状态特征与交互风险，输出继续维持还是放弃重选的二元决策。

### 2. 对照实验臂设计
1. **主实验臂（Learned Termination）**：学习型终止头，由反事实分叉标签监督训练。
2. **基准比较臂（LR Gated K=4）**：阶段 1 的逻辑回归门控（固定承诺 4 步）。
3. **新增对照臂（CUSUM Termination，累积和终止）**：
   - **机制设计**：不进行网络训练。在承诺存续期间的每一个决策步，计算“放弃并重选”相对于“继续维持当前承诺”的候选打分优势：
     $$D_t = J_{\text{abandon}} - J_{\text{continue}} = \max_a \text{Score}_{\text{base}}(a) - \left( \text{Score}_{\text{base}}(a_{\text{committed}}) + \alpha \cdot \text{compat}(a_{\text{committed}}, \text{intent}) \right)$$
   - **累积递推方程**：采用带泄漏的单侧累积和（Leaky CUSUM）递推累计：
     $$e_t = \text{clip}\left( \lambda \cdot e_{t-1} + (D_t - \Delta_{\text{floor}}) \cdot \Delta t, \; 0, \; e_{\max} \right)$$
     当累积量 $e_t \ge E_{\text{th}}$ 时，立即终止当前承诺，交由底层母体重新选择动作。
   - **参数选择规范**：参数三元组（泄漏系数 $\lambda$、判定门限 $E_{\text{th}}$、优势下限 $\Delta_{\text{floor}}$）严格只在 Case 99000–99399 的验证集部分（30% 案例，约 120 例）上进行有限网格搜索，选出一组最优参数；选定结果写入配置文件并提交 Git，然后再读取测试块。
   - **文献出处与简化说明**：
     - 出处：参照 COMPASS（Jungmo Kang, Preprints.org 2026, `doi:10.20944/preprints202610.0120.v1`）中的单侧累积量切换准则；
     - 简化：COMPASS 在连续轨迹层维护双人拓扑类别向量与几何进度硬化门限（$E_0(1+k_\rho \rho^p)$），本对照臂简化为对密集人群离散候选动作的打分优势进行累积，并使用常数门限 $E_{\text{th}}$ 与固定泄漏 $\lambda$。
4. **必带对照臂（Shuffled Labels Control）**：使用完全相同架构、特征与超参数，但在打乱标签（Shuffled Labels）下训练的伪终止头，结果与主读数并排列出。

### 3. 监督信号与反事实克隆验证
- 标签采集：在相同快照状态 $s_t$ 下，分叉两条续行分支：
  - 分支 A（Continue）：继续维持当前承诺意图；
  - 分支 B（Abandon）：放弃当前意图，交由底层母体重新选择动作/意图。
- 监督标签为两分支的未来累计回报差 $\Delta G = G_{\text{abandon}} - G_{\text{continue}}$。
- **克隆前置验证**：在启动分支前，验证无干预重放轨迹与原始环境轨迹逐步完全一致（100% Parity）。

### 4. 数据划分与候选配置
- 采样数据严格限制在 Case 99000 – 99399（400 例），按 Case ID 划分训练集（70%）与验证集（30%）。
- 候选模型配置事先列出（不超过 4 个候选），在验证集上依据验证损失与准确率选出唯一冠军配置，提交选定记录后再读取测试集。
- **回归闸门**：当终止头强制设为“第 4 步终止”时，其行为与结局必须与原逻辑回归门控（K=4）完全一致。

### 5. 测试块与双主比较判定判据
- **测试块**：Case 98000 – 98999（共 1000 例，Square Crossing，20 行人），只读取评估一次。
- **双主比较（Dual Primary Comparisons）**：
  必须同时通过以下两个比较，显著性水平各取 $\alpha = 0.025$（双侧精确 McNemar 检验）：
  1. **比较 1：可学习终止 对 逻辑回归门控（K=4）**
     - 成功率差：$\Delta\text{SR}_1 = \text{SR}_{\text{learned}} - \text{SR}_{\text{LR}}$
     - 双侧精确检验 $p_1$
  2. **比较 2：可学习终止 对 累积和终止（CUSUM）**
     - 成功率差：$\Delta\text{SR}_2 = \text{SR}_{\text{learned}} - \text{SR}_{\text{CUSUM}}$
     - 双侧精确检验 $p_2$
- **判定规则**：
  - **`SUPPORTED`**：
    两个比较均满足成功率差大于 0 且统计显著：
    $$(\Delta\text{SR}_1 > 0 \land p_1 < 0.025) \quad \land \quad (\Delta\text{SR}_2 > 0 \land p_2 < 0.025)$$
  - **`NOT SUPPORTED`**：
    任一比较的成功率差点估计小于 +1.0 个百分点：
    $$(\Delta\text{SR}_1 < +1.0\%) \quad \lor \quad (\Delta\text{SR}_2 < +1.0\%)$$
  - **`INCONCLUSIVE`**：
    其余所有情况（例如两个差均 $\ge +1.0\%$ 但未能均达到 $p < 0.025$）。
- **同步汇报指标**：
  同时报告对基线 B0 的差、碰撞率、超时率以及承诺持续步数（Dwell Steps）的经验分布。

---

## 三、资源与运行规范

1. **并行度限制**：在 `latent-type-headroom-20261014` 完成前，本任务最多使用 **4 个并行 worker 进程**，严禁触碰或干扰其进程与文件。
2. **客观报告规则**：
   - 阶段报告第一行只写状态词（`REPLICATED` / `NOT REPLICATED` / `INCONCLUSIVE`），不带任何修饰。
   - 所有数据均由自动化脚本直接从 JSON 计算生成，列出脚本与原始数据路径，严禁手填。
   - 禁用任何修饰性夸大词汇；不显著的差异严格不得称为“提升”。
   - 任何一臂成功率低于基线 10% 以上即视作异常实现故障，立刻停下排查。
