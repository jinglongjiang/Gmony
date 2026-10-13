# KDA-VL 三骨干训练与测评综合报告

更新日期：2026-10-07。状态：三臂训练、3000测评、续训至10,000及最终测评全部完成。

本报告只汇总本次原Mamba-VL项目的现有资产，不包含shixu旧实验，不新增训练或测评。
所有百分比由episode原始计数重新计算；pp表示百分点。

## 1. 先给结论

**本轮出现了一个值得复核的单seed正信号：官方KDA原位替换Mamba时序骨干，在累计10,000回合的同预算最终测评中，总体成功率为88.02%，Mamba和GRU均为85.94%。**

但这个结论必须同时带上以下事实：

- KDA只多成功4/192个episode，优势为2.08pp，不是大幅领先。
- 3000时KDA反而最差：83.33%，低于Mamba的88.54%和GRU的86.98%。不能删除这个负结果。
- 最终20人两种geometry合计，KDA成功50/64，其他两臂47/64；10人square中KDA仍不如Mamba。
- KDA总体碰撞15次，与Mamba相同，少于GRU的19次；不是所有场景都更安全。
- KDA参数较少，但本次完整动作评分的中位耗时略高，不能称为更快。
- 只有训练seed419，没有跨训练seed稳定性、fresh确认或机制归因证据。
- 这是“3000后统一重建replay再续训”的累计10,000，不是不间断10k；阶段变化不能全部归因于训练预算增加。

**当前裁决：原位KDA-VL跑通并获得单seed竞争力信号，值得固定配置复核；尚不足以宣布METHOD_ENTRY_FOUND、SCI方法成功或KDA普遍优于Mamba。**

## 2. 到底训练了什么

主项目：

/home/abc/workspace/nav_data/mamba/camrl/CrowdNav

```text
合法当前观测及历史
  -> 原34维状态编码
  -> 每帧[8,13] token
  -> 原Relational Spatial Encoder
  -> 24帧scene embedding，宽度256
  -> 4层temporal backbone：Mamba / GRU / 官方KDA
  -> 最后时刻Linear(256,1)
  -> 标量history-conditioned value
  -> 80个候选动作的一步后继评价：r + 0.99 V
  -> 原风险惩罚、安全过滤和执行合同
  -> 导航动作

训练：ORCA示范的MC价值回归初始化 -> 在线rollout -> MC return MSE更新
```

本次不是PPO，不是行为动作分类，不是预测器训练，也不是actor-memory、遮挡补全、forecast bridge或commit gate。没有从shixu搬入这些结构。

| 骨干 | 实际实现 | 共同输入输出 | 特别说明 |
|---|---|---|---|
| Mamba | mamba_ssm官方Mamba-1；4层，state64、conv4、expand2，原residual/LayerNorm | [B,24,256] -> [B,24,256] | 未安装可选causal-conv1d，使用官方库回退路径 |
| GRU | 4层、hidden256，配置dropout0.03，末端LayerNorm | 同上 | 继承的普通scene-level时序基线 |
| KDA | FLA官方KimiDeltaAttention；4层、4个64维head，原生q/k/v、短卷积、delta更新、通道decay、输出门控及projection | 同上 | 外层沿用residual/LayerNorm；不是完整Kimi Linear大模型 |

KDA每个窗口独立计算，past_key_values=None、use_cache=False；没有跨窗口外部持久状态、额外gate、辅助预测损失、MLA或MoE。value路径使用FP32，不使用autocast。

宽度、层数和输入输出匹配，不等于三种架构参数量相同；原生时序内部结构也并非逐项相同。

## 3. 公共合同与公平性

冻结基线19e6031，正式3000训练/评价来源b216efd，续训/评价来源1964f05。

| 核查项 | 本次实际合同 |
|---|---|
| 原项目继承 | CrowdNav、空间编码、scalar value、80动作、奖励、IL和online MC路径均来自原Mamba-VL |
| 字段修正 | robot radius/gx/gy/v_pref的解释与原始index一致；human x/y已是相对位置，不再重复减机器人坐标 |
| 评分与执行 | test/val候选先转换为实际平滑命令，再统一算successor、reward、clearance、value、filter、risk；不重复平滑 |
| train与test区别 | train仍执行未平滑grid动作，test执行平滑动作，这是继承的phase差异；不声称训练/评价行为完全一致 |
| 人数输入 | 先取供给顺序前5人，再在这些人内排序；不是从20人全局选最危险5人。reward/clearance仍检查全部行人 |
| 共享初始化 | seed419三臂所有非temporal模块hash完全一致；temporal采用各自原生初始化 |
| 配置 | 对两个预算阶段分别逐字段复核，三臂ini仅temporal_backbone不同；3000与10,000的ini仅train_episodes改变 |
| 在线数据 | 各臂自己采集在线轨迹，不能声称三臂online replay内容相同 |
| checkpoint选择 | 分别固定3000 final和10,000 final，没有使用中途最佳SR权重替换最终权重 |
| 非激活模块 | 原项目保留的Q/forecast头不参与本次scalar-value目标；参数总量包含这些继承的未激活头 |

共享初始化SHA256：

d515d50364ce850a110e44a7a62125be97b2deff80d8d251a973e017e6fbd71f

这些修正/回归是公共工程合同，不算KDA的新方法贡献。不能拿旧论文Mamba权重直接与修正后新KDA权重作公平主比较；本次三臂都在VL-v2下重新训练。

## 4. 数据、训练预算与评价协议

| 项目 | 实际值 |
|---|---|
| paired training seed | 419，只有一个训练seed |
| ORCA示范 | 同一原始池，15,000条成功示范，共749,318帧；5人训练域 |
| IL | 每臂50固定epochs，batch256，lr1e-4，AdamW；final epoch50，不回退best epoch |
| IL目标 | scalar Monte Carlo return MSE，不使用teacher动作编号分类 |
| RL | 先3000，再增加7000到累计10,000；仅5人circle训练 |
| RL更新 | batch256，lr1e-5，gamma0.99，每episode四次更新 |
| replay | capacity200,000、seq_len24；公共ORCA预填充5000条episode |
| exploration | epsilon0.30 -> 0.05，衰减1500回合；续训沿用累计episode编号，epsilon0.05 |
| 原奖励 | 成功+1；碰撞-0.5；超时-0.5；discomfort dist0.2、factor0.5；progress/time/stand额外奖励均为0 |
| 动作 | 5档指数采样速度 x 16方向，80动作，无stop动作；test平滑alpha0.3 |
| 测评 | 每臂5/10/20人 x circle/square x 32case = 192episodes |
| cases | 88000-88031，同一组ID用于六个cell；三个模型完全配对 |
| 环境 | circle半径4、square宽度10、原50秒limit、控制步0.25秒、到达半径0.25 |
| 评价策略 | epsilon0，加载对应final权重严格测评；不是训练rollout的滚动SR |

示范池SHA256：

940ae49af4b641d23b7c8bdf47f548dddce5db128b7ea0beaabe1e897459a8e2

teacher使用81动作grid，student使用80动作；本次IL为value回归，teacher action index不进入监督。这个差异保留记录，不能写成teacher/student动作支持完全相同。

3000及10,000各测576episodes，总计1152个episode结果；它们重复使用同一评价block，不是1152个独立新场景，也不是多个训练seed。原示范未记录可用于核对的全部case ID，不能严格证明评价case与示范池逐case无重叠。

### 4.1 续训的限制不能省略

原断点保存policy、target value network、AdamW、episode和统计历史，但没有保存在线replay、Python/NumPy/Torch RNG或环境case counter。

三臂真实日志均显示：

```text
IL episodes injected=5000, skipped=0, buffer_size=200000
Successfully resumed from episode 3001
restored_steps=[12000]
```

没有重做IL；原在线buffer丢失后统一用公共teacher池重建。最终优化器计数为40,000，意味着原12,000次更新之后确实增加28,000次更新。

**因此阶段间包含继续学习、replay重建、RNG/场景计数重置等共同变化。不能只凭KDA从83.33%到88.02%就断言“KDA需要10k才收敛”或“增加预算导致全部提升”。最终三臂遵循同一个续训协议，但不是原论文不间断10k的直接复现。**

## 5. 3000回合：完整保留的首轮结果

每臂192episodes，计数顺序为成功/碰撞/超时。

| 模型 | 成功/碰撞/超时 | SR | CR | TR | 平均原始折扣回报 |
|---|---|---:|---:|---:|---:|
| Mamba | 170/13/9 | 88.54% | 6.77% | 4.69% | 0.429635 |
| GRU | 167/14/11 | 86.98% | 7.29% | 5.73% | 0.445900 |
| KDA | 160/21/11 | 83.33% | 10.94% | 5.73% | 0.397960 |

| 场景，每格32episodes | Mamba成功/碰撞/超时 | GRU成功/碰撞/超时 | KDA成功/碰撞/超时 |
|---|---|---|---|
| 5人circle | 31/0/1 | 32/0/0 | 32/0/0 |
| 5人square | 32/0/0 | 32/0/0 | 31/0/1 |
| 10人circle | 30/2/0 | 29/1/2 | 26/4/2 |
| 10人square | 31/1/0 | 29/1/2 | 30/1/1 |
| 20人circle | 22/7/3 | 23/7/2 | 21/10/1 |
| 20人square | 24/3/5 | 22/5/5 | 20/6/6 |

已证实：在这一预算与这一block上，KDA总体落后且碰撞更多。这不能改写成“KDA一直领先”。

## 6. 累计10,000回合：主结果

| 模型 | 成功/碰撞/超时 | SR | CR | TR | 平均原始折扣回报 |
|---|---|---:|---:|---:|---:|
| Mamba | 165/15/12 | 85.94% | 7.81% | 6.25% | 0.448131 |
| GRU | 165/19/8 | 85.94% | 9.90% | 4.17% | 0.443802 |
| KDA | 169/15/8 | **88.02%** | 7.81% | 4.17% | **0.456664** |

KDA相对Mamba多4个成功、少4个超时，碰撞总数相同；相对GRU多4个成功、少4个碰撞，超时总数相同。

### 6.1 六场景明细

| 场景，每格32episodes | Mamba成功/碰撞/超时 | GRU成功/碰撞/超时 | KDA成功/碰撞/超时 | KDA SR |
|---|---|---|---|---:|
| 5人circle | 32/0/0 | 32/0/0 | 32/0/0 | 100.00% |
| 5人square | 31/1/0 | 32/0/0 | 32/0/0 | 100.00% |
| 10人circle | 28/1/3 | 29/3/0 | 30/1/1 | 93.75% |
| 10人square | **27/1/4** | 25/4/3 | 25/4/3 | 78.13% |
| 20人circle | 24/7/1 | 24/8/0 | 25/6/1 | 78.13% |
| 20人square | 23/5/4 | 23/4/5 | 25/4/3 | 78.13% |

**明确负面：10人square中，KDA比Mamba少2个成功、多3次碰撞，不能宣称各场景全面优越。** 每个cell只有32episode，一次结果变化就是3.125pp。

### 6.2 人数泛化与geometry分项

| 人数，两种geometry合计64episodes | Mamba SR | GRU SR | KDA SR | KDA相对Mamba/GRU |
|---|---:|---:|---:|---|
| 5人 | 98.44% | 100.00% | 100.00% | +1.56 / 0.00pp |
| 10人 | 85.94% | 84.38% | 85.94% | 0.00 / +1.56pp |
| 20人 | 73.44% | 73.44% | 78.13% | +4.69 / +4.69pp |

| geometry，三个人数合计96episodes | Mamba成功/碰撞/超时 | GRU成功/碰撞/超时 | KDA成功/碰撞/超时 |
|---|---|---|---|
| circle | 84/8/4 | 85/11/0 | 87/7/2 |
| square | 81/7/8 | 80/8/8 | 82/8/6 |

20人分项是本轮最值得确认的信号。但它仍只覆盖一个训练seed和一个32-ID block，且value只消费前5个人；不能外推为任意拥挤环境中的稳定泛化优势。

### 6.3 配对比较：既有挽救，也有破坏

下表逐episode匹配人数、geometry和case，不是仅比较两个独立均值。

| 对照 -> KDA，10,000权重 | 对照失败而KDA成功 | 对照成功而KDA失败 | 双方成功 | 成功净增 |
|---|---:|---:|---:|---:|
| Mamba -> KDA | 19：11碰撞、8超时被替换为成功 | 15：9变碰撞、6变超时 | 150 | +4 |
| GRU -> KDA | 20：14碰撞、6超时被替换为成功 | 16：10变碰撞、6变超时 | 149 | +4 |

这里“挽救”只是不同已训练模型在同case的结果差异，不是单模块介入的闭环因果实验。不能把19/20个正转换单独报告而隐藏15/16个负转换。

没有把这些episode当作独立训练种子做显著性声明。单seed、同case ID跨cell复用、配对得失接近，均限制证据强度。

## 7. 3000 -> 10,000变化

| 模型 | SR变化 | 碰撞变化 | 超时变化 | 原失败变成功 | 原成功变失败 |
|---|---|---|---|---:|---:|
| Mamba | 88.54 -> 85.94%，-2.60pp | 13 -> 15 | 9 -> 12 | 12 | 17 |
| GRU | 86.98 -> 85.94%，-1.04pp | 14 -> 19 | 11 -> 8 | 11 | 13 |
| KDA | 83.33 -> 88.02%，+4.69pp | 21 -> 15 | 11 -> 8 | 20 | 11 |

这说明评价表现并非随累计预算单调上升。Mamba总体SR下降同时平均原折扣回报上升；SR、回报和到达时间不是同一个指标。主比较仍使用相同预算final权重，不能事后选Mamba3000与KDA10,000作“公平领先”主表。

## 8. 训练是否真实发生、loss说明什么

本次重新读取6个正式RL权重，确认episode、policy有限性、optimizer state计数，以及权重SHA与对应evaluation.json一致：3000均step12,000，10,000均step40,000。

| 模型 | IL50训练MSE | RL500 | RL1000 | RL2000 | RL3000 | RL4000 | RL6000 | RL8000 | RL10,000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mamba | 0.000190403 | 0.0032 | 0.0033 | 0.0043 | 0.0039 | 0.0007 | 0.0009 | 0.0006 | 0.0005 |
| GRU | 0.000286965 | 0.0059 | 0.0051 | 0.0055 | 0.0032 | 0.0010 | 0.0010 | 0.0021 | 0.0009 |
| KDA | 0.000322531 | 0.0113 | 0.0087 | 0.0071 | 0.0061 | 0.0010 | 0.0022 | 0.0043 | 0.0017 |

RL数值来自各指定episode的[SARL-LOSS]日志，只有日志精度，属于抽样的训练minibatch MSE；不是独立测试value error，也不是80动作ranking error。三臂后续replay/target分布不同，不宜直接把loss大小等同于同一监督任务的泛化优劣。

IL Mamba/GRU的best_epoch元数据分别为35/40，仅是诊断记录；fixed-budget模式保存epoch50且不回退，不能误读成两臂IL实际只训35/40轮。

**已知：KDA最终导航SR更高，而其记录的RL10,000训练MSE仍更大。未知：这种差异来自怎样的表示、校准、轨迹分布或动作排名机制。当前数据不能证明KDA“拟合更好”“收敛更慢”或“矩阵检索能力带来收益”。**

## 9. 时间、回报、安全裕度与计算成本

### 9.1 导航耗时和间距

| 模型，10,000权重 | 成功episode平均到达时间 | 成功episode最小间距的中位数 | 该间距的P10 |
|---|---:|---:|---:|
| Mamba | 13.8652s | 0.198590m | 0.086158m |
| GRU | 13.5818s | 0.193316m | 0.073108m |
| KDA | 13.9127s | 0.197461m | 0.081993m |

不同模型成功子集不同，不能只凭这些条件均值认定KDA更慢或更快。仅在共同成功episode上配对：KDA-Mamba到达时间差均值-0.0417s，N=150；KDA-GRU为+0.4178s，N=149。

最小间距来自每个env.step之后的离散位置采样，扣除机器人和行人半径，不是连续时间的最小间距，也不是hard safety certificate。当前10,000的20人成功子集中，KDA间距中位数0.116242m，Mamba0.139410m、GRU0.120353m；不能把总体更少碰撞扩写成所有安全裕度都更大。

### 9.2 参数和完整80动作评分耗时

| 模型 | 模型总参数量 | 10,000测评predict中位耗时 | P95 |
|---|---:|---:|---:|
| Mamba | 2,223,337 | 29.97ms | 37.09ms |
| GRU | 1,754,857 | 29.70ms | 36.88ms |
| KDA | 1,506,809 | 31.68ms | 38.97ms |

总参数统计包含继承的未激活头。KDA总参数比Mamba少32.23%，比GRU少14.13%；这不等于活跃计算量同比降低。

耗时包括整条policy.predict、CPU部分和80候选value评分，前后CUDA同步，不是单KDA算子耗时。KDA在本次观测中并没有速度优势。GPU存在其他作业共享，Mamba缺少可选优化依赖，不能据此做严格架构速度排名或宣称“比最优化Mamba更高效”。

### 9.3 实际训练墙钟

| 模型 | IL50 + RL3000 | 续训7000回合 | 两阶段该臂耗时合计 |
|---|---:|---:|---:|
| Mamba | 4.058h | 3.484h | 7.542h |
| GRU | 3.808h | 3.768h | 7.576h |
| KDA | 4.634h | 4.361h | 8.995h |

第一阶段逐臂执行，含预处理和原生诊断评价；续训三臂并行，含重建buffer及诊断评价。不能把三臂合计时长当实际总墙钟，也不能排除GPU/CPU资源共享影响。

续训共同启动时刻为2026-10-07 02:48:50（Asia/Seoul）。Mamba于06:17:54、GRU于06:34:56、KDA于07:10:31被控制器记录完成；轮询检测有最多约15秒误差。最终本地镜像记录在07:32:32同步完成并退出。

3000训练的采样GPU总峰值：Mamba17,076MiB、GRU12,300MiB、KDA15,966MiB，包含同卡其他任务及分配器保留空间，**不能用于每模型独占VRAM结论**。本次没有合格的独占显存/算子效率benchmark。

## 10. 工程过程与负结果保留

| 阶段 | 实际问题/操作 | 当前处理及证据边界 |
|---|---|---|
| 初次smoke | 缺少动作合同所需的同级env.config | 补齐原配置，不改grid；失败发生在有效训练前 |
| replay核查 | SARL选中动作的grid index返回None，在线episode存储失败 | 公共metadata修正；实际动作和目标定义不变；不推断历史论文replay状况 |
| KDA100 smoke | raw-init、无IL，100episode；541.65s，有限权重和非空optimizer | 只验证技术稳定，不把此SR当正式比较；smoke权重未用于正式三臂 |
| IL预处理 | 原逐重叠窗口转换重复计算，首次尝试在优化前暂停 | 改为逐帧转换再组相同窗口；fixture与32条真实轨迹对比bitwise一致 |
| IL固定预算 | 首次flag未正确通过TrainConfig wrapper，优化前失败 | 修正参数传递；三臂50epoch，不按结果选择best |
| 评价加载 | torch.compile导致_orig_mod.名称前缀 | 只去掉前缀，strict加载原tensor |
| 续训 | 原权重没有online replay/RNG/env counter | 三臂统一重建teacher replay，不重IL，明确不是不断点复现 |
| 正确性回归 | 索引、MC target、IL窗口/预算、实际CLI续训 | 本地和服务器四项训练合同测试通过；另保留冻结骨干/reference/causal等检查资产 |

所有失败尝试和日志保留在3000 artifact目录。没有根据3000排名修改KDA结构、奖励、输入或optimizer；本次续训只改变累计RL预算及明确记录的统一断点恢复流程。

## 11. 哪些结论已经成立，哪些没有

| 证据层级 | 可以说什么 | 不能说什么 |
|---|---|---|
| 已证实事实 | 三臂同一原项目pipeline实际完成IL、online MC及两轮权重测评；final SHA核对通过 | 只有forward测试、没有训练 |
| 已证实事实 | 单seed419、既定评价block上，10,000 KDA总体SR领先2.08pp；20人领先4.69pp | 四seed一致、所有cell更优、泛化已经稳定 |
| 已证实事实 | KDA参数少，观测predict耗时略高；3000负结果和成功破坏存在 | KDA已经在速度/安全所有维度支配基线 |
| 有证据支持的推断 | 该scene-level官方KDA原位用法至少具有竞争力，值得一次固定协议复核 | 旧shixu所有负结果均无效，或当前KDA已经满足论文贡献 |
| 尚未证明 | 跨seed/fresh稳健性、历史长度效应、channel-wise decay的机制增量 | 把这些当本轮发现 |
| 尚未证明 | 续训改善的唯一原因、表示/critic/历史如何导致具体case改变 | 通过总SR反推根因或“已解决某种时序缺陷” |
| 尚未证明 | 新颖性与SCI完整实验充分性 | “首次使用KDA”自动构成方法创新 |

本轮与原论文87.9%的数值不能直接比较：公共合同修正、当前小评价block和续训方式均有区别，且未完成论文六场景各500episode的正式复现。数值接近不等于复现成功。

## 12. 下一步建议与停止边界

**唯一优先建议：保持当前模型与公共合同不动，用新的paired training seeds和预先冻结的fresh评价block复核三臂差分。现在不是再加gate、actor memory、forecast或commitment。**

本报告不自动启动这些工作。执行前应明确选择同一续训协议，或为正式实验统一建立可完整恢复replay/RNG/env counter的训练合同；不能混用旧Mamba不断训练和新KDA断点重建数据。

若后续至少3/4 paired seeds和fresh block都保留对应的成功/安全收益，再进入KDA原生机制消融及论文组织。若优势不重复、强GRU/Mamba已吸收，或安全损害抵消收益，就撤回当前增量主张，而不是靠测试集调参救结果。

现在支持“继续复核”，不支持“已找到SCI方法”，也不支持因3000落后就判KDA整个架构无效。

## 13. 可复核资产

综合报告：

/home/abc/workspace/nav_data/mamba/camrl/CrowdNav/KDA-VL-COMPREHENSIVE-REPORT.md

3000原始协议、evaluation、训练日志、全部权重：

/home/abc/workspace/nav_data/mamba/camrl/CrowdNav/artifacts/vl-3000-screening

10,000原始协议、evaluation、训练日志、全部权重：

/home/abc/workspace/nav_data/mamba/camrl/CrowdNav/artifacts/vl-10000-continuation

公共合同：VL-v2-contract.md。冻结正确性资产：artifacts/kda-vl-freeze。两阶段summary为各自comparison.json；逐episode细节在各模型evaluation.json。

| 核心代码，相对主项目路径 | 本轮用途 |
|---|---|
| crowd_nav/train.py | 原IL、MC在线训练及严格续训；不是新训练范式 |
| crowd_nav/contracts.py | 原始字段到token、公共动作grid合同 |
| crowd_nav/policy/mamba_rl.py | 空间编码、Mamba/GRU、scalar value、80动作评分及实际命令 |
| crowd_nav/policy/kda_temporal.py | 薄封装官方FLA KimiDeltaAttention；无手写近似替代 |
| crowd_nav/policy/shared_initialization.py | paired seed的非temporal共享初始化 |
| tools/run_vl_minimal.py | 固定50IL/3000预算、最终权重测评 |
| tools/resume_vl_parallel.py | 先完成3000评价，再三臂并行续训，最后评价10,000 |
| tools/test_vl_training_contract.py | 工程合同回归，不作为导航性能证据 |

### 权重SHA256

| 模型 | 3000 final | 10,000 final |
|---|---|---|
| Mamba | 960f4f17abe4f34adddc1f6c0a0bef312e573697dc34f84bc9397faf83ec2abe | fe04e811c3bf9f88757f3825a65879c8748307078afbd686472c3dc4780e192f |
| GRU | 77e7bc2f706f1b04710d1deb86f0063513c245c742809479aeaa33f4678c4cd1 | ba8f3276421e5672479ea3fec20fe01dd6d53c8a4ac9bd0fb662423fd14376eb |
| KDA | 55acb5c08a3c62b78ef9fe92477ffda42be16e07c23b2dc2af5218bbc5e2909f | 138ed2e46fbb676c8dd0c62c3b9575360349fbc6c0f6ab868f89d8c8d1e0445b |

本次未重新开展文献/novelty审计，也未新增模型训练、导航episode或消费者。报告的导航数字完全来自已经保存的权重测评。
