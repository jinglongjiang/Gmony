# 阶段 3 文献综述：动作坚持、选项终止与人群导航承诺机制的技术核查

本综述聚焦核心科学问题：**“在保留底层逐步重规划的前提下，学习何时启动和终止对一个高层意图的偏置”**，对已有领域中最接近的三类工作进行逐条文献核实与机制差异分析。

---

## 1. 带终止学习的选项方法 (Option Methods with Learned Termination)

### 1.1 Option-Critic 架构
- **论文信息**：*The Option-Critic Architecture*
- **作者**：Pierre-Luc Bacon, Jean Harb, Doina Precup
- **发表年份与出处**：2017 年，Proceedings of the 31st AAAI Conference on Artificial Intelligence (AAAI-17), pp. 1726–1734.
- **链接**：[arXiv:1609.05140](https://arxiv.org/abs/1609.05140) / [AAAI 出版物页面](https://ojs.aaai.org/index.php/AAAI/article/view/10916)
- **阅读状态**：已阅读全文。
- **原文核心机制与实验范围**：
  在半马尔可夫决策过程（SMDP）框架下，推导了内部策略梯度定理（Intra-Option Policy Gradient Theorem）与终止梯度定理（Termination Gradient Theorem），使智能体能够利用标准环境回报端到端同时学习 Option 内部动作策略 $\pi_\omega(a \mid s)$、Option 价值函数 $Q_\Omega(s, \omega)$ 与终止函数 $\beta_\omega(s)$。在离散四房间网格环境与连续控制/Atari 游戏中进行了验证。
- **与本方案在可检验层面上的具体差别**：
  1. **底层重规划与安全约束的存在性**：Option-Critic 原文中，每个激活的 Option 通过其内部策略网络 $\pi_\omega(a \mid s)$ 直接输出底层的原子控制动作（例如网格移动或关节扭矩），底层不保留任何运动学规划器或基于物理间隙的一步价值前瞻。本方案完全保留底层母体规划器（`score_sarl_candidates`）在每一决策步（0.25 秒）对 80 个运动候选动作的评价、TTC5 行人几何扫掠与安全距离强制剔除，高层意图仅体现为加在候选动作得分上的偏置项（Score Bias），不剥夺底层的避障与重规划能力。
  2. **终止函数的学习信号**：Option-Critic 的终止函数由策略梯度根据 TD 误差 $Q_\Omega(s, \omega) - V_\Omega(s)$ 进行无监督端到端联合更新；本方案在阶段 2 中是将终止建模为基于同状态精确反事实分叉回报差（$\Delta G = G_{\text{abandon}} - G_{\text{continue}}$）监督的有监督二元决策头。

### 1.2 引入协商成本的 Option-Critic
- **论文信息**：*When Waiting Is Not an Option: Learning Options with a Deliberation Cost*
- **作者**：Jean Harb, Pierre-Luc Bacon, Martin Klissarov, Doina Precup
- **发表年份与出处**：2018 年，Proceedings of the 32nd AAAI Conference on Artificial Intelligence (AAAI-18), pp. 3165–3172.
- **链接**：[arXiv:1709.04571](https://arxiv.org/abs/1709.04571) / [AAAI 出版物页面](https://ojs.aaai.org/index.php/AAAI/article/view/11649)
- **阅读状态**：已阅读全文。
- **原文核心机制与实验范围**：
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
- **链接**：[arXiv:2106.05262](https://arxiv.org/abs/2106.05262) / [PMLR 出版物页面](http://proceedings.mlr.press/v139/biedenkapp21a.html)
- **阅读状态**：已阅读全文。
- **原文核心机制与实验范围**：
  在 MDP 中引入状态跳跃连接（Skip Connections），在基线动作策略之外额外训练一个跳跃策略 $\pi_{\text{skip}}(k \mid s)$，使智能体在选择动作 $a_t$ 的同时决定该动作的重复执行步数 $k \in \{1, \dots, K\}$。在跳跃期间跳过中间状态的策略计算，直接重复执行相同的物理动作。在经典控制基准环境与深度强化学习任务中验证了训练加速效果。
- **与本方案在可检验层面上的具体差别**：
  1. **重复的对象不同**：TempoRL 重复的是**同一个底层原子物理动作** $a_t$。在执行跳跃的 $k$ 步内，底层规划器处于休眠状态，智能体不感知中间环境变化，无法根据突发行人调整航向或刹车；本方案**严禁重复底层原子动作**，每个决策步底层依然实时感知全部行人并计算 80 个速度/航向候选，重复或坚持的仅是“偏好左绕/直行/右绕/减速”的高层拓扑偏置。
  2. **启动时机不同**：TempoRL 在每一个决策点都由跳跃策略输出一个持续步数 $k$；本方案平时完全由底层反应式规划器主导，仅在检测到明显的局部停滞或振荡前兆（位移比超标、速度频繁反向）且物理安全间隙脱离紧急危险区时，才触发意图门控介入。

### 2.2 细粒度动作重复 (FiGAR)
- **论文信息**：*Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning*
- **作者**：Sahil Sharma, Aravind S. Lakshminarayanan, Balaraman Ravindran
- **发表年份与出处**：2017 年，5th International Conference on Learning Representations (ICLR 2017).
- **链接**：[arXiv:1702.06054](https://arxiv.org/abs/1702.06054) / [OpenReview 页面](https://openreview.net/forum?id=B1kJ6H9ex)
- **阅读状态**：已阅读全文。
- **原文核心机制与实验范围**：
  提出 Fine Grained Action Repetition 框架，联合输出离散动作与该动作的重复频次，在 Atari 2600、MuJoCo 连续控制和 TORCS 赛车模拟器上结合 A3C、TRPO 和 DDPG 进行了实验，验证自适应动作重复对探索与计算效率的改善。
- **与本方案在可检验层面上的具体差别**：
  FiGAR 同样是在执行周期内对底层低阶控制信号进行硬性锁存（Zero-Order Hold），跳过中间环境交互与规划。本方案在承诺持续期间每一帧均执行完整的运动学前瞻与碰撞风险筛选，绝不进行底层开环锁存。

---

## 3. 人群导航中的通过侧承诺与迟滞机制 (Passing Commitment & Hysteresis in Crowd Navigation)

### 3.1 社会动量框架 (Social Momentum)
- **论文信息**：*Social Momentum: A Framework for Legible Navigation in Dynamic Multi-Agent Environments*
- **作者**：Christoforos I. Mavrogiannis, Wil B. Thomason, Ross A. Knepper
- **发表年份与出处**：2018 年，Proceedings of the 2018 ACM/IEEE International Conference on Human-Robot Interaction (HRI 2018), pp. 361–369.
- **链接**：[ACM Digital Library (DOI: 10.1145/3171221.3171280)](https://dl.acm.org/doi/10.1145/3171221.3171280)
- **阅读状态**：已阅读全文。
- **原文核心机制与实验范围**：
  针对人机相向通行中的“互易舞蹈（Reciprocal Dance）”振荡现象，提出社会动量（Social Momentum）优化框架，将行人和机器人的交互模式投影到拓扑编织（Topological Braids）空间，通过在目标泛函中最大化运动意图的可读性（Legibility），促使机器人维持明确的绕行侧，避免左右摇摆。在受控室内场景与仿真多智能体交叉环境中进行了主客观评测。
- **与本方案在可检验层面上的具体差别**：
  1. **拓扑建模的扩展性**：原文依赖对每对智能体轨迹的连续编织同伦类分析，适用于结构化通道或少数几人的成对交互；在 20 人全向交叉的密集随机流中，全局拓扑同伦类维度呈组合爆炸且快速退化。本方案通过局部 TTC5 关系将空间简化为左绕、直突、右绕、减速 4 个宏观方向类。
  2. **学习方式**：原文通过解析拓扑泛函的数值优化求解轨迹；本方案是通过反事实仿真数据直接训练轻量级门控与终止判别器，学习何时给打分加偏置、何时撤销。

### 3.2 基于累积量承诺的通行规划器 (COMPASS)
- **论文信息**：*COMPASS: Temporally Consistent Topological Passing Decisions for Socially Aware Robot Navigation*
- **作者**：Jungmo Kang
- **发表年份与出处**：2026 年，Preprints.org 预印本，manuscript 202610.0120/v1.
- **链接**：[Preprints.org 页面](https://www.preprints.org/manuscript/202610.0120/v1)
- **阅读状态**：只读了摘要与引言/公开导览部分。
- **原文核心机制与实验范围**：
  针对局部反应式规划器在行人近身时频繁改变通过侧的震荡问题，提出 COMPASS 框架。利用带有泄漏项的单侧 CUSUM（累积和）统计检验算法构建决策层承诺，绑定行人 Track ID，提供任意两次意图切换之间的理论步数下界，并在检测到突发安全风险时允许优先级覆写。在仿真与实物移动机器人通过场景中评测了切换次数抑制效果。
- **与本方案在可检验层面上的具体差别**：
  1. **决策准则的形式**：COMPASS 采用解析推导的单侧 CUSUM 统计检验与人工设定的证据泄漏率/进度阈值；本方案采用数据驱动方式，基于反事实分叉标签学习逻辑回归/随机森林分类器，直接学习干预收益的运行边界。
  2. **终止机制**：COMPASS 的终止主要依赖统计累积量越过下阈值或到达几何通过线；本方案阶段 2 采用同状态克隆分叉的回报差作为显式监督信号，训练轻量终止头学习意图撤销时机。

---

## 4. 总结：本方案在技术坐标系中的定位

通过对上述三类工作的核查，本方案的技术边界与已有工作的本质差异可精炼为以下两点：
1. **与 Option-Critic 及 TempoRL 的区别**：本方案**不剥夺底层规划器的逐帧重规划、一步前瞻与硬几何安全过滤能力**，亦不硬性锁存低阶控制指令；高层意图仅作为加在候选动作得分上的**软性偏置项（Soft Score Bias）**。
2. **与人群导航承诺/迟滞工作的区别**：已有工作主要依赖人工设计的固定几何距离、同伦类泛函或解析统计量（如 CUSUM），且多针对单人或稀疏通道；本方案在 20 人密集交叉场景中，利用**同状态反事实分叉生成的回报增益**作为监督信号，通过轻量级分类器学习意图偏置的**启动门控边界**与**可学习终止时机**。
