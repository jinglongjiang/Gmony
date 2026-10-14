# 离线执行与阶段 2 状态监控报告 (STATUS_WHILE_AWAY)

> 更新时间：2026-10-15 01:35:00  
> 工作分支：`research/option-critic-cf-10h`  
> 运行资源更正说明：此前第一版闸门（PID 1574308，2 进程）启动早于单线程重构，实测各占约 365% CPU（未受单线程限制，此前 STATUS 概括为"全部单线程"与事实不符，在此更正）。已遵照指示终止旧闸门，原 10 例结果归档为 `null_branch_gate_partial_v1.json`；已为 `null_branch_gate.py` 补充完整单线程限制（环境变量 + `torch.set_num_threads(1)`）并重新启动。数据采集 6 进程始终严格单线程（每进程 CPU 78%–81%）。

---

## 一、每一步的状态总览

| 阶段 / 任务 | 状态 | 简要说明 |
| :--- | :--- | :--- |
| **第 1 步：立即执行与冒烟测试** | **已验证通过** | 旧采集已杀；修复 `log_f` NameError 并提交；冒烟测试 99000–99001 成功通过（单线程 97.5%–98.3% CPU、逐 case 刷新日志、逐 case 写入均验证通过，已提交） |
| **第 2 步：空分叉闸门（Case 99000–99019）** | **已验证通过 (PASSED)** | 2 进程单线程重跑全部 20 例已全量完成；540/540 分叉步全部通过（max_act_diff=0.00e+00, max_g_diff=0.00e+00）；与 partial_v1 前 10 例逐例核验 100% 精确一致 |
| **第 3 步：阶段 2 正式数据采集（99000–99399）** | **已验证通过 (全量完成)** | 6 个 worker 进程全部正常完成（退出码 0），总耗时 25,965.9s (~7.2h)，完成全部 400 个 Case，产出 9,783 个分叉样本并完成合并与统计分析 |
| **第 4 步：CUSUM 调参与终止头训练** | **已验证通过 (全量完成)** | CUSUM 调参完成（8 候选评测，胜出者 cusum_cand1_fast，SR 80.83% 已提交）；终止头 4 候选模型训练及验证块闭环评测完成，胜出者 cand4_mlp_h32_16 (SR 85.83%, CR 6.67%)；打乱标签对照组闭环评测完成 (SR 88.33%, CR 6.67%)；ΔSR=-2.50%, ΔCR=0.00%, ΔTime=-0.37s；全量模型与元数据已提交；测试块 98000–98999 严格未运行 |

---

## 二、所有 Git 提交号汇总

1. [`ffe52e2`](https://github.com/jinglongjiang/Gmony/commit/ffe52e2) - `stage2: add capture_internal_state and restore_internal_state to policy and implement null branch gate`
2. [`93e34fe`](https://github.com/jinglongjiang/Gmony/commit/93e34fe) - `stage2: refactor collect_stage2_data to use policy.restore_internal_state and add recommitment tracking`
3. [`ae69a1b`](https://github.com/jinglongjiang/Gmony/commit/ae69a1b) - `stage2: re-run forced step 4 regression parity gate across 20 cases (100% parity passed, 0.00e+00 diff)`
4. [`a87c6ad`](https://github.com/jinglongjiang/Gmony/commit/a87c6ad) - `stage2: configure null_branch_gate for workers=2 with real-time incremental saving`
5. [`b2a2454`](https://github.com/jinglongjiang/Gmony/commit/b2a2454) - `stage2: add per-case flush logging, per-case incremental save, and single-thread constraints to collect_stage2_data`
6. [`6096157`](https://github.com/jinglongjiang/Gmony/commit/6096157) - `stage2: enforce single-thread constraints across tuning, training, and evaluation scripts`
7. [`71e2f85`](https://github.com/jinglongjiang/Gmony/commit/71e2f85) - `stage2: fix NameError by restoring log_f file opening before subprocess launch`
8. [`ee8aadb`](https://github.com/jinglongjiang/Gmony/commit/ee8aadb) - `stage2: verify smoke test on cases 99000-99001 (per-case flush log, incremental save, single-thread 98% CPU verified)`
9. [`ed2ea55`](https://github.com/jinglongjiang/Gmony/commit/ed2ea55) - `stage2: allow up to 6 workers for stage2 collection`
10. [`025f971`](https://github.com/jinglongjiang/Gmony/commit/025f971) - `stage2: create STATUS_WHILE_AWAY.md for autonomous monitoring`
11. [`f9e8fa7`](https://github.com/jinglongjiang/Gmony/commit/f9e8fa7) - `stage2: record 15-minute collection milestone in STATUS_WHILE_AWAY.md`
12. [`f9dc6cc`](https://github.com/jinglongjiang/Gmony/commit/f9dc6cc) - `stage2: record 30-minute monitoring row in STATUS_WHILE_AWAY.md`
13. [`b1af868`](https://github.com/jinglongjiang/Gmony/commit/b1af868) - `stage2: record 60-minute monitoring row (44/400 cases, 1175 samples, case 99010 passed)`
14. [`ed35c54`](https://github.com/jinglongjiang/Gmony/commit/ed35c54) - `stage2: record 90-minute monitoring row (64/400 cases, 1962 samples, gate running past 90m per protocol)`
15. [`d48dafb`](https://github.com/jinglongjiang/Gmony/commit/d48dafb) - `stage2: record 120-minute monitoring row (94/400 cases, 2757 samples, 6 workers healthy)`
16. [`853c248`](https://github.com/jinglongjiang/Gmony/commit/853c248) - `stage2: record 150-minute monitoring row (139/400 cases, 3716 samples, 45 new cases)`
17. [`acb35ad`](https://github.com/jinglongjiang/Gmony/commit/acb35ad) - `stage2: record 180-minute monitoring row (174/400 cases, 4319 samples, 43.5% done)`
18. [`e714719`](https://github.com/jinglongjiang/Gmony/commit/e714719) - `stage2: record 210-minute monitoring row (211/400 cases, 5332 samples, 52.8% done)`
19. [`c399a6c`](https://github.com/jinglongjiang/Gmony/commit/c399a6c) - `stage2: record 240-minute monitoring row (245/400 cases, 5959 samples, 61.3% done)`
20. [`ac457c8`](https://github.com/jinglongjiang/Gmony/commit/ac457c8) - `stage2: record 270-minute monitoring row (279/400 cases, 6590 samples, 69.8% done)`
21. [`2702030`](https://github.com/jinglongjiang/Gmony/commit/2702030) - `stage2: record 300-minute monitoring row (315/400 cases, 7509 samples, 78.8% done)`
22. [`9c68968`](https://github.com/jinglongjiang/Gmony/commit/9c68968) - `stage2: enforce single-thread constraints in null_branch_gate.py, archive partial_v1 results, and correct resource statement in STATUS`
23. [`fab5311`](https://github.com/jinglongjiang/Gmony/commit/fab5311) - `stage2: record 10m gate CPU check (99% single-core, 7/7 cases match partial_v1) and 348m collection milestone (347/400 cases)`
24. [`9ff707b`](https://github.com/jinglongjiang/Gmony/commit/9ff707b) - `stage2: null branch gate PASSED across all 20 cases (540/540 steps, diff 0.00e+00, 100% matched partial_v1)`
25. [`b9744e9`](https://github.com/jinglongjiang/Gmony/commit/b9744e9) - `stage2: add test block (98000, 98999) to FORBIDDEN_RANGES in tune_cusum.py and train_stage2_termination.py`
26. [`9b7a188`](https://github.com/jinglongjiang/Gmony/commit/9b7a188) - `stage2: save formal branching dataset (9783 samples across 400 cases) and stats`
27. [`f23faaa`](https://github.com/jinglongjiang/Gmony/commit/f23faaa) - `stage2: add flush logging and 4-worker multiprocessing to tune_cusum and train_stage2_termination, update STATUS with dataset stats`
28. [`3fae424`](https://github.com/jinglongjiang/Gmony/commit/3fae424) - `stage2: commit CUSUM grid search results and selected parameters on validation split`

---

## 三、空分叉闸门（Null Branch Gate）实时明细

- **测试案例范围**：Case 99000 – 99019（共 20 例）
- **判定阈值**：容差 $10^{-9}$，动作序列与折现回报必须逐步完全相等
- **单线程重跑验证**：以 2 进程单线程（OMP/MKL/OPENBLAS 环境变量 + `torch.set_num_threads(1)`）重新运行，启动 10 分钟实测 CPU：PID 1688315 @ 98.8%、PID 1688317 @ 99.1%（严格符合单线程要求，彻底消除此前 365% 争用）。全量 20 例总耗时 3781.99s（~63 分钟）。
- **partial_v1 逐例比对**：前 10 例（99000, 99001, 99002, 99003, 99005, 99006, 99007, 99008, 99009, 99010）与归档的 `null_branch_gate_partial_v1.json` 逐例、逐步 100% 精确一致，无任何浮点或离散漂移。
- **最终结论**：**已验证通过 (PASSED)**。全部 20 例 540 个分叉步 100% 通过（`max_act_diff = 0.00e+00`, `max_g_diff = 0.00e+00`）。

### 实时案例比对数据表

| Case ID | 回合总步数 | 承诺分支步数 | 耗时 (s) | `max_act_diff` | `max_g_diff` | 状态 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 99000 | 75 | 27 | 303.4 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99001 | 59 | 16 | 141.1 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99002 | 121 | 12 | 234.2 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99003 | 61 | 12 | 87.0 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99004 | 201 | 106 | 2212.8 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99005 | 55 | 4 | 43.5 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99006 | 64 | 8 | 83.7 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99007 | 68 | 16 | 136.8 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99008 | 55 | 4 | 43.4 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99009 | 70 | 18 | 120.3 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99010 | 84 | 38 | 261.8 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99011 | 201 | 164 | 2493.5 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99012 | 201 | 28 | 770.5 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99013 | 66 | 12 | 104.7 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99014 | 55 | 7 | 49.5 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99015 | 74 | 20 | 154.2 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99016 | 65 | 14 | 94.1 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99017 | 62 | 15 | 115.0 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99018 | 21 | 0 | 3.4 | 0.00e+00 | 0.00e+00 | **已验证通过 (无承诺步)** |
| 99019 | 81 | 19 | 108.0 | 0.00e+00 | 0.00e+00 | **已验证通过** |

### 最终 Summary 原文（`replication/null_branch_gate.json`）
```json
{
  "gate_name": "Stage 2 Null Branch Replay Gate",
  "status": "PASSED",
  "total_cases": 20,
  "completed_cases": 20,
  "case_range": [
    99000,
    99019
  ],
  "total_sampled_steps": 540,
  "passed_sampled_steps": 540,
  "all_passed": true,
  "tolerance": 1e-09,
  "max_return_diff": 0.0,
  "max_action_diff": 0.0,
  "max_robot_pos_diff": 0.0,
  "max_reward_diff": 0.0,
  "elapsed_seconds": 3781.99
}
```

---

## 四、数据采集进度与速度跟踪表

> 监控规则：每 30 分钟检查各分卷文件与日志并追加记录。
> 实测基准（冒烟测试实测）：Case 99000 耗时 195.1s（27 样本，7.2s/样本），Case 99001 耗时 77.8s（16 样本，4.9s/样本），单线程平均约 136.5s/Case。
> 6 个 worker 承担 400 个 Case（约 67 Case/worker），总期望时长推算：$67 \times 136.5\text{s} \approx 9,145\text{s} \approx 2.54\text{小时}$。

| 记录时间 | 运行墙上时长 | 已完成 Case 数 | 累计采集样本数 | 实测均速 (s/Case) | 剩余时间估算 | 进程健康状态 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 16:40:34 | 0m | 0 / 400 | 0 | *(启动中)* | 约 2.5～3.0 小时 *(推测)* | 6 个进程全部正常运行，单核 CPU 85%–88% |
| 16:55:34 | 15m | 23 / 400 | 453 | 39.1s/Case (综合墙上耗时) | 约 3.8～4.1 小时 (实测吞吐 1.53 Case/min) | 6 个进程全部已打印 2～7 个 Case 进度，0 报错，逐 Case 存盘正常 |
| 17:10:45 | 30m | 29 / 400 | 559 | 62.1s/Case (综合墙上耗时) | 约 6.4 小时 (实测累计吞吐 0.967 Case/min) | 6 个进程健康运行（CPU 85%–88%），0 报错，磁盘剩余 47GB |
| 17:42:01 | 60m | 44 / 400 | 1175 | 83.9s/Case (综合墙上耗时) | 约 8.3 小时 (实测累计吞吐 0.715 Case/min) | 6 个进程健康运行（CPU 85%–88%），0 报错，磁盘剩余 47GB |
| 18:13:00 | 92m | 64 / 400 | 1962 | 86.7s/Case (综合墙上耗时) | 约 8.1 小时 (实测累计吞吐 0.692 Case/min) | 6 个进程健康运行（CPU 78%–81%），0 报错，磁盘剩余 47GB |
| 18:45:00 | 124m | 94 / 400 | 2757 | 79.3s/Case (综合墙上耗时) | 约 6.7 小时 (实测累计吞吐 0.757 Case/min) | 6 个进程健康运行（CPU 78%–81%），0 报错，磁盘剩余 47GB |
| 19:16:00 | 155m | 139 / 400 | 3716 | 67.1s/Case (综合墙上耗时) | 约 4.9 小时 (实测累计吞吐 0.894 Case/min) | 6 个进程健康运行（CPU 78%–81%），0 报错，磁盘剩余 47GB |
| 19:47:30 | 187m | 174 / 400 | 4319 | 64.5s/Case (综合墙上耗时) | 约 4.1 小时 (实测累计吞吐 0.930 Case/min) | 6 个进程健康运行（CPU 78%–81%），0 报错，磁盘剩余 47GB |
| 20:18:50 | 218m | 211 / 400 | 5332 | 62.0s/Case (综合墙上耗时) | 约 3.3 小时 (实测累计吞吐 0.968 Case/min) | 6 个进程健康运行（CPU 78%–81%），0 报错，磁盘剩余 47GB |
| 20:50:00 | 249m | 245 / 400 | 5959 | 61.1s/Case (综合墙上耗时) | 约 2.6 小时 (实测累计吞吐 0.982 Case/min) | 6 个进程健康运行（CPU 78%–81%），0 报错，磁盘剩余 47GB |
| 21:21:25 | 281m | 279 / 400 | 6590 | 60.4s/Case (综合墙上耗时) | 约 2.0 小时 (实测累计吞吐 0.994 Case/min) | 6 个进程健康运行（CPU 78%–81%），0 报错，磁盘剩余 47GB |
| 21:52:40 | 312m | 315 / 400 | 7509 | 59.4s/Case (综合墙上耗时) | 约 1.4 小时 (实测累计吞吐 1.009 Case/min) | 6 个进程健康运行（CPU 78%–81%），0 报错，磁盘剩余 47GB |
| 22:28:50 | 348m | 347 / 400 | 8298 | 60.2s/Case (综合墙上耗时) | 约 0.9 小时 (实测累计吞吐 0.997 Case/min) | 6 个进程健康运行（CPU 78%–81%），0 报错，磁盘剩余 47GB |
| 23:53:20 | 432m | 400 / 400 | 9783 | 64.9s/Case (综合墙上耗时) | 0 小时 (已全量完成) | 6 个进程全部退出码 0 正常完成，合并为 stage2_branching_dataset.json |

---

## 五、阶段 2 分叉数据集统计明细 (N=9,783)

### 1. 划分分布 (Split Distribution)
- **样本总数**：9,783 个分叉样本（覆盖 Case 99000 – 99399 共 400 个 Case）
- **训练块 (Cases 99000 – 99279, N=280)**：6,584 样本 (67.30%)
- **验证块 (Cases 99280 – 99399, N=120)**：3,199 样本 (32.70%)
- **测试块 (Cases 98000 – 98999)**：严格禁止接触，0 样本

### 2. 放弃分支优于继续分支比例 ($G_{abandon} > G_{continue}$, 正样本率)
- **全集总体**：4,800 / 9,783 = **49.06%**（正负样本高度均衡）
- **训练块**：3,044 / 6,584 = **46.23%**
- **验证块**：1,756 / 3,199 = **54.89%**
- **回报差 $\Delta G = G_{abandon} - G_{continue}$ 分布**：
  - 均值 (Mean)：+0.0768
  - 标准差 (Std)：0.1937
  - 中位数 (Median)：0.0000
  - 极值范围：[-1.1506, +1.4298]

### 3. 按 Case 的样本与正样本率分布
- 400 个 Case 中，产生承诺分叉样本的 Case：**355 / 400 (88.8%)**
- 未产生承诺分叉样本的 Case（即全程未触发承诺门控或仅单步）：**45 / 400 (11.2%)**
- 在 355 个有效 Case 中的样本量分布：
  - 最小值：1
  - 第 25 百分位数 (p25)：11.0
  - 中位数 (Median)：18.0
  - 第 75 百分位数 (p75)：29.5
  - 第 90 百分位数 (p90)：54.0
  - 最大值：179
  - 均值 $\pm$ 标准差：27.56 $\pm$ 31.39
- 各 Case 放弃分支占优比例（正样本率）：
  - 跨 Case 平均正样本率：28.30%
  - 仅继续更优（正样本率 = 0.0%）：73 个 Case (20.6%)
  - 仅放弃更优（正样本率 = 100.0%）：31 个 Case (8.7%)
  - 混合正负样本（0.0% < 正样本率 < 100.0%）：251 个 Case (70.7%)

### 4. 采样时已保持步数 (Dwell Step) 分布
- `dwell_step = 1`：3,558 样本 (36.4%) | 放弃更优：1,987 (55.8%)
- `dwell_step = 2`：2,483 样本 (25.4%) | 放弃更优：1,348 (54.3%)
- `dwell_step = 3`：2,009 样本 (20.5%) | 放弃更优：996 (49.6%)
- `dwell_step = 4`：1,733 样本 (17.7%) | 放弃更优：469 (27.1%)

### 5. 放弃后重新承诺动态 (Re-commitment Dynamics)
- 放弃后在回合内重新承诺：9,029 / 9,783 (**92.3%**)
- 重新承诺选择**相同意图**：4,426 (49.0%)
- 重新承诺选择**不同意图**：4,603 (51.0%)
- 重新承诺间隔步数：均值 2.26 步，中位数 1.0 步（极值范围 1 ~ 86 步）

---

## 六、CUSUM 参数网格调优结果 (Cases 99280–99399, N=120)

- **评估方法**：4 个单线程 worker 并行评估 8 组候选参数，严格隔离于验证块（99280–99399），禁区拦截生效。
- **总耗时**：3,193.4s (~53.2 分钟)
- **产物文件**：[`replication/cusum_selected_params.json`](file:///home/abc/workspace/autonomous_research_20261012/replication/cusum_selected_params.json)

| 候选名称 | decay | threshold | floor | 闭环 SR (%) | 闭环 CR (%) | 闭环 TR (%) | 平均耗时 (s) | 平均 Dwell |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **cusum_cand1_fast (Champion)** | **0.80** | **0.10** | **0.00** | **80.83%** | **8.33%** | **10.83%** | **18.19** | **4.11** |
| cusum_cand2_med_low | 0.85 | 0.15 | 0.00 | 80.83% | 8.33% | 10.83% | 18.19 | 4.11 |
| cusum_cand3_med | 0.90 | 0.20 | 0.00 | 80.83% | 8.33% | 10.83% | 18.19 | 4.11 |
| cusum_cand4_med_floor | 0.90 | 0.15 | 0.05 | 80.83% | 8.33% | 10.83% | 18.19 | 4.11 |
| cusum_cand5_slow | 0.95 | 0.25 | 0.00 | 80.83% | 8.33% | 10.83% | 18.19 | 4.11 |
| cusum_cand6_slow_floor | 0.95 | 0.20 | 0.05 | 80.83% | 8.33% | 10.83% | 18.19 | 4.11 |
| cusum_cand7_conservative | 0.98 | 0.35 | 0.05 | 80.83% | 8.33% | 10.83% | 18.19 | 4.11 |
| cusum_cand8_persistent | 1.00 | 0.30 | 0.10 | 80.83% | 8.33% | 10.83% | 18.19 | 4.11 |

*注：8 组参数结果相同，因多数步数中未受限最佳动作与受限动作一致（$\text{score\_gap} \approx 0$），CUSUM 均由安全硬上限或紧急避障释放触发。*

---

## 七、阶段 2 可学习终止头训练与打乱标签对照组结果

### 1. 候选模型筛选（训练集 Cases 99000–99279，验证块 Cases 99280–99399 闭环评选）

| 候选模型 | 离线 Acc (%) | 离线 AUC | 离线 F1 | 闭环验证 SR (%) | 闭环验证 CR (%) | 闭环验证 TR (%) | 闭环验证时间 (s) | 平均 Dwell |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cand1_lr_c1.0` (逻辑回归 C=1.0) | 72.71% | 0.8092 | 0.7141 | 81.67% | 7.50% | 10.83% | 17.95s | 2.03 |
| `cand2_lr_c0.1` (逻辑回归 C=0.1) | 72.74% | 0.8093 | 0.7143 | 82.50% | 7.50% | 10.00% | 18.08s | 2.07 |
| `cand3_rf_d4_n50` (随机森林 d=4) | 73.12% | 0.8281 | 0.7037 | 80.00% | 9.17% | 10.83% | 18.12s | 2.18 |
| **`cand4_mlp_h32_16` (MLP 冠军)** | **72.21%** | **0.8155** | **0.7267** | **85.83%** | **6.67%** | **7.50%** | **17.79s** | **2.14** |

### 2. 真实标签冠军 vs 打乱标签对照组（Shuffled Label Control Arm）对比与差值

对照组采用与 Champion 完全相同的 MLP 网络结构（隐层 32, 16），在训练集特征上拟合随机打乱的标签（随机置换种子 42），并在完全相同的验证块（120 例）上进行闭环仿真评测。

| 指标项 | 真实标签 Champion (`cand4_mlp_h32_16`) | 打乱标签对照组 (Shuffled Control) | 差值 ($\Delta = \text{Real} - \text{Shuffled}$) |
| :--- | :---: | :---: | :---: |
| **离线验证 Acc** | 72.21% | 50.92% | **+21.29%** |
| **离线验证 AUC** | 0.8155 | 0.5477 | **+0.2678** |
| **离线验证 F1** | 0.7267 | 0.4120 | **+0.3147** |
| **闭环成功率 (SR)** | **85.83%** | **88.33%** | **-2.50%** |
| **闭环碰撞率 (CR)** | **6.67%** | **6.67%** | **+0.00%** |
| **闭环超时率 (TR)** | **7.50%** | **5.00%** | **+2.50%** |
| **导航耗时 (Time)** | **17.79s** | **18.16s** | **-0.37s** |
| **平均保持步数 (Dwell)** | **2.14** | **2.02** | **+0.12 步** |

### 3. 产物提交
- 元数据文件：[`replication/stage2_champion_metadata.json`](file:///home/abc/workspace/autonomous_research_20261012/replication/stage2_champion_metadata.json)
- 模型权重：[`replication/stage2_champion_model.joblib`](file:///home/abc/workspace/autonomous_research_20261012/replication/stage2_champion_model.joblib)
- 对照组模型：[`replication/stage2_shuffled_model.joblib`](file:///home/abc/workspace/autonomous_research_20261012/replication/stage2_shuffled_model.joblib)
- 测试块（Cases 98000–98999）：**严格未运行，做到第 4 步立即停机等待指示**。
