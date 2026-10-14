# 阶段 3 文献综述：动作坚持、意图偏置与人群导航承诺机制的技术核查

本综述聚焦核心科学问题：**“在保留底层逐步重规划的前提下，学习何时启动和终止对一个高层意图的偏置”**，对已有领域中最接近的研究工作进行逐条文献核查、机制剖析与可检验层面的差异分析。

---

## 1. 带终止学习的选项方法 (Option Methods with Learned Termination)

### 1.1 Option-Critic 架构
- **论文信息**：*The Option-Critic Architecture*
- **作者**：Pierre-Luc Bacon, Jean Harb, Doina Precup
- **发表年份与出处**：2017 年，Proceedings of the 31st AAAI Conference on Artificial Intelligence (AAAI-17), pp. 1726–1734.
- **权威链接**：[arXiv:1609.05140](https://arxiv.org/abs/1609.05140) / [AAAI 出版物页面](https://ojs.aaai.org/index.php/AAAI/article/view/10916)
- **阅读状态**：已阅读全文。
- **原文核心机制与验证域**：
  在半马尔可夫决策过程（SMDP）框架下，推导了内部策略梯度定理（Intra-Option Policy Gradient Theorem）与终止梯度定理（Termination Gradient Theorem），使智能体能够利用环境标量回报端到端联合学习 Option 内部动作策略 $\pi_\omega(a \mid s)$、Option 价值函数 $Q_\Omega(s, \omega)$ 与终止函数 $\beta_\omega(s)$。原文在离散四房间网格与连续控制/Atari 游戏中进行了验证。
- **与本方案在可检验层面上的具体差别**：
  1. **底层重规划与安全约束的存在性**：Option-Critic 原文中，每个激活的 Option 通过其内部策略网络 $\pi_\omega(a \mid s)$ 直接输出底层的原子控制动作（例如网格移动或关节扭矩），底层不保留任何运动学规划器或基于物理间隙的一步价值前瞻。本方案完全保留底层母体规划器（`score_sarl_candidates`）在每一决策步（0.25 秒）对 80 个运动候选动作的评价、TTC5 行人几何扫掠与安全距离强制剔除，高层意图仅体现为加在候选动作得分上的偏置项（Score Bias），不剥夺底层的避障与重规划能力。
  2. **终止函数的学习信号**：Option-Critic 的终止函数由策略梯度根据 TD 误差 $Q_\Omega(s, \omega) - V_\Omega(s)$ 进行无监督端到端联合更新；本方案在阶段 2 中是将终止建模为基于同状态精确反事实分叉回报差（$\Delta G = G_{\text{abandon}} - G_{\text{continue}}$）监督的有监督二元决策头。

### 1.2 引入协商成本的 Option-Critic
- **论文信息**：*When Waiting Is Not an Option: Learning Options with a Deliberation Cost*
- **作者**：Jean Harb, Pierre-Luc Bacon, Martin Klissarov, Doina Precup
- **发表年份与出处**：2018 年，Proceedings of the 32nd AAAI Conference on Artificial Intelligence (AAAI-18), pp. 3165–3172.
- **权威链接**：[arXiv:1709.04571](https://arxiv.org/abs/1709.04571) / [AAAI 出版物页面](https://ojs.aaai.org/index.php/AAAI/article/view/11649)
- **阅读状态**：已阅读全文。
- **原文核心机制与验证域**：
  针对标准 Option-Critic 中 Option 容易退化为单步切换的现象，基于有限理性（Bounded Rationality）理论引入常数协商成本 $\eta > 0$（Deliberation Cost），在终止梯度中对重新选择 Option 的行为施加惩罚，促使网络学习在时间上持续更长的 Option。在 Arcade Learning Environment (ALE) 的多款 Atari 游戏中进行了实验。
- **与本方案在可检验层面上的具体差别**：
  1. **动作执行与意图偏置模式**：原文依然遵循由 Option 策略网络直接输出游戏控制指令的框架，缺乏底层基于预测模型的局部重规划与动态安全边界约束。
  2. **切换控制机制**：原文使用固定标量 $\eta$ 作为全环境统一的切换成本惩罚项；本方案并不使用全局标量正则化，而是在可观测特征空间中学习成本敏感型分类门控（阶段 1）与反事实收益判别器（阶段 2），按状态局部动态决定何时介入与何时退出。

---

## 2. 动作时长与跳跃重复学习 (TempoRL 及其后续)

### 2.1 TempoRL: 学习何时行动
- **论文信息**：*TempoRL: Learning When to Act*
- **作者**：André Biedenkapp, Raghu Rajan, Frank Hutter, Marius Lindauer
- **发表年份与出处**：2021 年，Proceedings of the 38th International Conference on Machine Learning (ICML 2021), PMLR 139:914–924.
- **权威链接**：[arXiv:2106.05262](https://arxiv.org/abs/2106.05262) / [PMLR 出版物页面](http://proceedings.mlr.press/v139/biedenkapp21a.html)
- **阅读状态**：已阅读全文。
- **原文核心机制与验证域**：
  在 MDP 中引入状态跳跃连接（Skip Connections），在基线动作策略之外额外训练一个跳跃策略 $\pi_{\text{skip}}(k \mid s)$，使智能体在选择动作 $a_t$ 的同时决定该动作的重复执行步数 $k \in \{1, \dots, K\}$。在跳跃期间跳过中间状态的策略计算，直接重复执行相同的物理动作。在经典控制基准环境与深度强化学习任务中验证了训练加速效果。
- **与本方案在可检验层面上的具体差别**：
  1. **重复的对象不同**：TempoRL 重复的是**同一个底层原子物理动作** $a_t$。在执行跳跃的 $k$ 步内，底层规划器处于休眠状态，智能体不感知中间环境变化，无法根据突发行人调整航向或刹车；本方案**严禁重复底层原子动作**，每个决策步底层依然实时感知全部行人并计算 80 个速度/航向候选，重复或坚持的仅是“偏好左绕/直行/右绕/减速”的高层拓扑偏置。
  2. **启动时机不同**：TempoRL 在每一个决策点都由跳跃策略输出一个持续步数 $k$；本方案平时完全由底层反应式规划器主导，仅在检测到明显的局部停滞或振荡前兆（位移比超标、速度频繁反向）且物理安全间隙脱离紧急危险区时，才触发意图门控介入。

### 2.2 细粒度动作重复 (FiGAR)
- **论文信息**：*Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning*
- **作者**：Sahil Sharma, Aravind S. Lakshminarayanan, Balaraman Ravindran
- **发表年份与出处**：2017 年，5th International Conference on Learning Representations (ICLR 2017).
- **权威链接**：[arXiv:1702.06054](https://arxiv.org/abs/1702.06054) / [OpenReview 页面](https://openreview.net/forum?id=B1kJ6H9ex)
- **阅读状态**：已阅读全文。
- **原文核心机制与验证域**：
  提出 Fine Grained Action Repetition 框架，联合输出离散动作与该动作的重复频次，在 Atari 2600、MuJoCo 连续控制和 TORCS 赛车模拟器上结合 A3C、TRPO 和 DDPG 进行了实验，验证自适应动作重复对探索与计算效率的改善。
- **与本方案在可检验层面上的具体差别**：
  FiGAR 同样是在执行周期内对底层低阶控制信号进行硬性锁存（Zero-Order Hold），跳过中间环境交互与规划。本方案在承诺持续期间每一帧均执行完整的运动学前瞻与碰撞风险筛选，绝不进行底层开环锁存。

---

## 3. COMPASS 深度全文核查 (基于作者源码与论文手稿)

- **论文信息**：*COMPASS: Temporally Consistent Topological Passing Decisions for Socially Aware Robot Navigation*
- **作者**：Jungmo Kang
- **发表年份与出处**：2026 年，Preprints.org 预印本，manuscript 202610.0120/v1；开源实现位于 GitHub 仓库 `kjungmo/compass`。
- **权威链接**：[Preprints.org 页面](https://doi.org/10.20944/preprints202610.0120.v1) / [GitHub 仓库](https://github.com/kjungmo/compass) / [论文手稿全文源文件](https://raw.githubusercontent.com/kjungmo/compass/main/paper/paper_draft.md)
- **阅读状态**：**已取得并通读全文（Full Text）**。

基于通读全文手稿（`paper/paper_draft.md`）、LaTeX 源码（`paper/arxiv/main.tex`）及 C++ 核心实现（`src/compass_core`），针对四项核心机制的核查结论如下：

### 3.1 承诺对象 (Commitment Target)
COMPASS 的承诺对象是**绑定到行人 Track ID 的离散拓扑通过类别向量**：
$$c = (s_{id_1}, \dots, s_{id_K}), \quad s \in \{L, R\}$$
即针对局部感知的 $K$ 位相关行人（典型 $K=1 \sim 3$ 人），由左侧通过（$L$）或右侧通过（$R$）符号组成一个拓扑类别标签，至多产生 $2^K$ 种可能组合。系统通过维护行人 Track ID 的生命周期（生成、消灭、合并、拆分及 TTL 记忆）在不同决策步之间追踪同一拓扑意图。

### 3.2 启动和终止的判定方式 (Initiation & Termination Criteria)
1. **启动（初始承诺建立）**：在交互发生时，由加权综合社会代价函数 $J(c) = w_g J_{\text{goal}} + w_s J_{\text{social}} + w_e J_{\text{effort}} + w_r J_{\text{rule}}$ 对候选类别进行评估，选取当前无偏置最优的可用类别初始化 $c^*$，累积量初始化为 $e \leftarrow 0$。
2. **终止与切换（从当前承诺 $c^*$ 退出并切换到竞争类别 $c'$）**：
   - 计算挑战类别 $c'$ 相对于当前承诺 $c^*$ 的优势值：$D_k = J_k(c^*_{k-1}) - J_k(c'_k)$；
   - 采用带有泄漏项的单侧 CUSUM（累积和）递推方程积分证据：
     $$e_k = \text{clip}\left( \lambda e_{k-1} + (D_k - \Delta_{\text{floor}}) \Delta t, \; 0, \; e_{\max} \right)$$
   - **终止并切换门限**：仅当累积证据超过随执行进度硬化的门限时才允许退出当前承诺并切换：
     $$e_k \ge E_{\text{th}}(\rho_k) = E_0 \left(1 + k_\rho \rho_k^p\right)$$
     其中 $\rho_k \in [0, 1]$ 表示沿当前承诺路径已执行的横向位移比率（Progress Hardening / Point of No Return）。
3. **安全覆写终止（Lexicographic Safety Override）**：
   若当前承诺 $c^*$ 在某步变为不可行（`Safe_k(c*) == False`，即违反硬安全距离 $d_{\text{safe}}$ 或 TTC 门限），系统立即绕过证据累积方程（以零迟滞），直接执行安全降级阶梯：寻找安全替代类别 $\to$ 减速制动 $\to$ 若滑动时间窗内安全介入次数达到 $N_{\text{thrash}}$ 则强制进入 `HOLD` 零速度停滞状态。

### 3.3 底层是否逐步重规划 (Step-by-step Replanning)
**是，底层保持严格的逐帧重规划。**
COMPASS 在架构上分为三层：决定层（Decision Layer）、轨迹层（Trajectory Layer）和表达层（Expression Layer）：
- 决定层（`DecisionCore`）仅负责维持离散拓扑类别 $c^*$；
- 决定层输出给底层控制器后，在每个控制周期（例如 20 Hz，$\Delta t = 0.05\,\text{s}$），底层 `nav2_core::Controller` 插件依据实时的 2D 代价地图与行人恒速外推预测，在被选定的拓扑通道内重新计算连续的底盘速度指令（$v, \omega$），或者执行 1.0 秒单轮驱动前瞻（unicycle rollout）重新评估并下发第一步速度。底层并不锁死控制指令。

### 3.4 实验的人数规模和结果 (Human Scale & Empirical Results)
1. **人数规模**：
   - 原文的离线仿真评测（Offline Harness）主要针对**成对近身遭遇（Pairwise Encounters，典型为 $K=1 \sim 3$ 名行人）**；
   - **未开展 20 人全向密集交叉流（如 CrowdNav 经典 Square Crossing）等大尺度群智交互环境下的闭环仿真**。原文在 §5.6 中明确声明：物理闭环评估、与外部基线的对比以及实人用户实验在本文中尚未实施，仅作为计划中的评估协议列出。
2. **原文实验结果**：
   - 在 5 个复现遭遇场景（模棱两可微弱差 `near_tie`、瞬态峰值 `transient_spike`、中途逆转 `mid_reversal`、明显优势 `clean_commit`、间歇优势 `intermittent`）$\times 50$ 随机种子（共 250 次离线测试）中：
     - 即时贪婪 argmin 策略在模棱两可遭遇下平均每 10 秒发生 **98.16 次** 决策方向翻转，全场景平均每遭遇 **29.74 次** 翻转；
     - COMPASS 凭借累积量门限在模棱两可遭遇下决策切换为 **0 次**，全场景平均每遭遇仅 **0.40 次** 切换（最多 1 次切换），大幅消除了由于噪声导致的左右摇摆（Chattering）。
     - 核心决策计算延迟（$K=3$ 即 8 个类别）在 AMD Ryzen 5 5500GT 上 $p99$ 为 $58.6\,\mu\text{s}$，最大 $651.3\,\mu\text{s}$，占 20 Hz 控制周期的 1.30%。

---

## 4. 扩展检索：三类相关技术与可检验差异

### 4.1 学习在多个策略或规划器之间切换或仲裁的方法 (Learning to Switch / Arbitrate Planners)

- **代表文献**：*APPLD: Adaptive Planner Parameter Learning from Demonstration*
- **作者**：Xuesu Xiao, Bo Liu, Garrett Warnell, Jonathan Fink, Peter Stone
- **发表年份与出处**：2020 年，IEEE Robotics and Automation Letters (RA-L 2020), 5(3): 4541–4548.
- **权威链接**：[arXiv:2004.00116](https://arxiv.org/abs/2004.00116) / [IEEE Xplore (DOI: 10.1109/LRA.2020.3002219)](https://doi.org/10.1109/LRA.2020.3002219)
- **阅读状态**：已阅读全文。
- **原文核心机制**：
  提出自适应规划器参数学习框架（APPLD）。在移动机器人运行过程中，并不替代经典局部规划器（如 DWA、E-Band），而是利用示范数据训练情境分类器，根据局部环境传感器上下文（如开阔走廊、狭窄门口、转角），在线动态切换底层规划器的全局超参数集（如最大线速度、目标吸引权重、障碍物代价权重）。
- **与本方案在可检验层面上的具体差别**：
  1. **调节对象不同**：APPLD 切换的是底层规划器的**全局行为超参数**（Parameter Tuning），对所有候选运动施加全局缩放；本方案并不更改规划器的运动学参数与评估准则，而是在动作打分空间为特定拓扑方向（宏观意图）施加**方向性打分偏置**。
  2. **切换依据不同**：APPLD 依赖静态环境几何的特征聚类；本方案的切换门控专注于动态人群博弈中产生的**交互死锁与振荡特征**（路径位移比、速度反向频次），并且门控的判定边界由反事实分叉标签直接监督。

### 4.2 在基础规划器上叠加学习型修正的方法 (Superimposing Learned Corrections on Base Planners)

- **代表文献**：*Residual Reinforcement Learning for Robot Control*
- **作者**：Tobias Johannink, Shikhar Bahl, Ashvin Nair, Jianlan Luo, Avinash Kumar, Matthias Loskyll, Juan Aparicio Ojea, Sergey Levine
- **发表年份与出处**：2019 年，IEEE International Conference on Robotics and Automation (ICRA 2019), pp. 6007–6013.
- **权威链接**：[arXiv:1812.03201](https://arxiv.org/abs/1812.03201) / [IEEE Xplore (DOI: 10.1109/ICRA.2019.8794125)](https://doi.org/10.1109/ICRA.2019.8794125)
- **阅读状态**：已阅读全文。
- **原文核心机制**：
  提出残差强化学习（Residual RL）架构。将最终控制信号显式分解为经典反馈控制器输出与强化学习策略输出的直接叠加：$u = u_{\text{base}} + u_{\text{residual}}$。传统控制器处理显式建模的宏观结构，RL 策略专注于补偿非线性摩擦与接触等残差项。在机器人精密装配任务中验证了有效性。
- **与本方案在可检验层面上的具体差别**：
  1. **叠加空间与安全保证机制不同**：残差 RL 在**动作输出空间（Action Space）**直接相加物理控制量（力矩或速度）。这种直接相加容易破坏底层传统控制器原有的硬性边界约束（例如碰撞不变量与执行器限制）。本方案是在**候选动作打分空间（Score Space）**进行软偏置叠加；底层规划器针对 80 个候选动作计算的几何扫掠间隙与安全阈值过滤位于偏置之后，若偏置后的最高分动作违反安全间隙，依然会被底层硬性剔除，保证底层安全包络不被破坏。
  2. **时间连续性与介入方式不同**：残差 RL 在每个控制步都持续输出微调残差；本方案平时残差偏置为 0，仅由成本敏感分类器在检测到死锁时离散地激活 $K$ 步意图偏置。

### 4.3 专门处理人群导航中冻结或振荡的方法 (Mitigating Freezing and Oscillations in Crowd Navigation)

- **代表文献 1**：*Unfreezing the Robot: Navigation in Dense, Interacting Crowds*
  - **作者**：Peter Trautman, Andreas Krause
  - **发表年份与出处**：2010 年，IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2010), pp. 797–803.
  - **权威链接**：[IEEE Xplore (DOI: 10.1109/IROS.2010.5654369)](https://doi.org/10.1109/IROS.2010.5654369) / [Caltech 官方主页 PDF](https://www.vision.caltech.edu/publications/trautman-krause-iros-2010.pdf)
  - **阅读状态**：已阅读全文。
  - **原文核心机制**：
    首次形式化定义了人群导航中的“冻结机器人问题（Freezing Robot Problem）”：指出传统方法将行人视作被动移动障碍物，在高密度人群中由于不确定性膨胀导致规划器认为无路可走而停滞。提出了交互高斯过程（Interactive Gaussian Process, IGP），通过显式建模人机协同避让（Cooperative Collision Avoidance）来解冻机器人。
  - **与本方案在可检验层面上的具体差别**：
    Trautman 的方法属于解析交互模型，假设所有行人都具备对称的协同让行倾向；而在非完全合作或混合人群中，无条件假设人机协同可能在行人不让行时造成安全隐患。本方案不改变底层的保守碰撞模型，而是针对“因机器人过度退让导致停滞”的特定工况，通过门控有针对性地对通行方向注入偏置。

- **代表文献 2**：*Hybrid Reciprocal Velocity Obstacles for Cooperative Multi-Agent Collision Avoidance*
  - **作者**：Jamie Snape, Jur van den Berg, Stephen J. Guy, Dinesh Manocha
  - **发表年份与出处**：2011 年，IEEE Transactions on Robotics (IEEE T-RO 2011), 27(4): 696–706.
  - **权威链接**：[IEEE Xplore (DOI: 10.1109/TRO.2011.2120810)](https://doi.org/10.1109/TRO.2011.2120810) / [UNC 官方页面](https://gamma.cs.unc.edu/HRVO/)
  - **阅读状态**：已阅读全文。
  - **原文核心机制**：
    针对交互速度障碍物（RVO）与最佳互易速度障碍物（ORCA）在相向通行时出现的“互易舞蹈（Reciprocal Dance）”振荡问题，提出了混合互易速度障碍物（HRVO）。通过在速度障碍物几何构造中将尖端朝一侧非对称倾斜，打破几何对称性，使得双方在博弈中自然选择同一侧避让，从解析几何层面消除振荡。
  - **与本方案在可检验层面上的具体差别**：
    HRVO 是一种纯反应式的底层速度几何障碍物修剪算法，必须依赖所有智能体执行相同的 HRVO 协议；在混杂行人的非对称非结构化人群中，行人不会遵守机器人的几何 HRVO 规则。本方案工作在单智能体分层决策层面，在对方行人策略未知的情况下，通过学习型门控在宏观意图层注入持续性偏置以克服振荡。

---

## 5. 总结：本方案在技术坐标系中的定位

综上所述，已有文献在各个单点上均有探索，但本方案的技术切入点具有清晰的定位：
1. **相对 Option-Critic 与 TempoRL**：本方案**不去除底层逐步重规划，不放弃扫掠几何安全过滤，亦不对低阶物理动作进行硬锁存**；持续承诺仅作用于高层拓扑打分偏置。
2. **相对 COMPASS 与 Social Momentum**：本方案克服了解析同伦类在 20 人密集交叉流中的组合爆炸局限，**基于反事实仿真数据直接学习正负收益的运行边界**，而非依赖人工调校的累积量参数或固定几何线。
3. **相对 APPLD 与 Residual RL**：本方案不在全局参数层（APPLD）或连续控制量输出层（Residual RL）做粗粒度叠加，而是在**候选动作打分空间进行带物理间隙衰减的安全有界偏置**。
