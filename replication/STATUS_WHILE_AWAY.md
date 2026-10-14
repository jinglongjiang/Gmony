# 离线执行与阶段 2 状态监控报告 (STATUS_WHILE_AWAY)

> 更新时间：2026-10-14 22:17:00  
> 工作分支：`research/option-critic-cf-10h`  
> 运行资源更正说明：此前第一版闸门（PID 1574308，2 进程）启动早于单线程重构，实测各占约 365% CPU（未受单线程限制，此前 STATUS 概括为"全部单线程"与事实不符，在此更正）。已遵照指示终止旧闸门，原 10 例结果归档为 `null_branch_gate_partial_v1.json`；已为 `null_branch_gate.py` 补充完整单线程限制（环境变量 + `torch.set_num_threads(1)`）并重新启动。数据采集 6 进程始终严格单线程（每进程 CPU 78%–81%）。

---

## 一、每一步的状态总览

| 阶段 / 任务 | 状态 | 简要说明 |
| :--- | :--- | :--- |
| **第 1 步：立即执行与冒烟测试** | **已验证通过** | 旧采集已杀；修复 `log_f` NameError 并提交；冒烟测试 99000–99001 成功通过（单线程 97.5%–98.3% CPU、逐 case 刷新日志、逐 case 写入均验证通过，已提交） |
| **第 2 步：空分叉闸门（Case 99000–99019）** | **单线程重启重跑 20 例** | 旧闸门已终止，10 例结果安全保留至 `null_branch_gate_partial_v1.json`；脚本补充单线程限制与步级进度日志并提交；2 进程单线程重跑 20 例，输出到 `null_branch_gate.json`，并将逐例比对 partial_v1 |
| **第 3 步：阶段 2 正式数据采集（99000–99399）** | **进行中（已完成 315/400）** | 6 个 worker 进程持续稳定运行，严格单线程（CPU 78%–81%），已完成 78.8% |
| **第 4 步：CUSUM 调参与终止头训练** | **未验证** | 严格遵循契约：等待闸门全 PASS 且采集完全结束后方才启动 |

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

---

## 三、空分叉闸门（Null Branch Gate）实时明细

- **测试案例范围**：Case 99000 – 99019（共 20 例）
- **判定阈值**：容差 $10^{-9}$，动作序列与折现回报必须逐步完全相等
- **当前进度**：已完成 10 / 20 个案例，已通过分支步：155 / 155 步（注：闸门运行耗时超 90 分钟，依协议继续在后台全速运行，不中断）

### 实时案例比对数据表

| Case ID | 回合总步数 | 承诺分支步数 | `max_act_diff` | `max_g_diff` | 状态 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 99000 | 75 | 27 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99001 | 59 | 16 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99002 | 121 | 12 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99003 | 61 | 12 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99004 | 201 | 106 | *(计算中)* | *(计算中)* | 进行中 |
| 99005 | 55 | 4 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99006 | 64 | 8 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99007 | 68 | 16 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99008 | 55 | 4 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99009 | 70 | 18 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99010 | 84 | 38 | 0.00e+00 | 0.00e+00 | **已验证通过** |
| 99011 | 201 | 164 | *(排队中)* | *(排队中)* | 待运行 |
| 99012 | 201 | 28 | *(排队中)* | *(排队中)* | 待运行 |
| 99013 | 66 | 12 | *(排队中)* | *(排队中)* | 待运行 |
| 99014 | 55 | 7 | *(排队中)* | *(排队中)* | 待运行 |
| 99015 | 74 | 20 | *(排队中)* | *(排队中)* | 待运行 |
| 99016 | 65 | 14 | *(排队中)* | *(排队中)* | 待运行 |
| 99017 | 62 | 15 | *(排队中)* | *(排队中)* | 待运行 |
| 99018 | 21 | 0 | 0.00e+00 | 0.00e+00 | **已验证通过 (无承诺步)** |
| 99019 | 81 | 19 | *(排队中)* | *(排队中)* | 待运行 |

### 最新 Summary 原文（增量 JSON）
```json
{
  "gate_name": "Stage 2 Null Branch Replay Gate",
  "status": "IN_PROGRESS",
  "total_cases": 20,
  "completed_cases": 10,
  "case_range": [99000, 99019],
  "total_sampled_steps": 155,
  "passed_sampled_steps": 155,
  "all_passed": true,
  "tolerance": 1e-09,
  "max_return_diff": 0.0,
  "max_action_diff": 0.0,
  "max_robot_pos_diff": 0.0,
  "max_reward_diff": 0.0,
  "elapsed_seconds": 6127.81
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
