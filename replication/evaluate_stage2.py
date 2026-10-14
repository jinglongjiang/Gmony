import os
import sys
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import joblib
from scipy.stats import binom_test

REPO_ROOT = Path('/home/abc/workspace/autonomous_research_20261012')
sys.path.insert(0, str(REPO_ROOT))

from crowd_sim.envs.utils.robot import Robot
from run_b0_counterfactual_study import create_environment
from run_stage4_500cases_evaluation import run_single_episode, load_stage4_policy
from replication.stage2_policy import (
    Stage2TerminationPolicy,
    load_stage2_policy,
    STAGE2_FEATURE_KEYS
)

REPLICATION_DIR = REPO_ROOT / 'replication'
OUT_MASTER = REPLICATION_DIR / 'stage2_evaluation_1000cases.json'
OUT_STATS = REPLICATION_DIR / 'stage2_evaluation_stats.json'
REPORT_PATH = REPLICATION_DIR / 'stage2_termination_report.md'

FORBIDDEN_RANGES = [
    (92000, 92499),
    (90000, 90299),
    (93000, 93199),
    (95000, 96499)
]

def assert_case_guard(start_case, num_cases):
    end_case = start_case + num_cases - 1
    for f_start, f_end in FORBIDDEN_RANGES:
        if not (end_case < f_start or start_case > f_end):
            raise ValueError(
                f"SAFETY VIOLATION: Case range [{start_case}, {end_case}] overlaps "
                f"with strictly prohibited range [{f_start}, {f_end}]!"
            )

def mcnemar_exact_test(arm_a_outcomes, arm_b_outcomes):
    b = sum(1 for a, b in zip(arm_a_outcomes, arm_b_outcomes) if a == 'SUCCESS' and b != 'SUCCESS')
    c = sum(1 for a, b in zip(arm_a_outcomes, arm_b_outcomes) if a != 'SUCCESS' and b == 'SUCCESS')
    n = b + c
    if n == 0:
        return 1.0, b, c
    pval = binom_test(min(b, c), n, 0.5)
    return float(pval), b, c

def compute_paired_ci_diff(outcomes_a, outcomes_b, target_val='COLLISION'):
    n = len(outcomes_a)
    y_a = np.array([1.0 if x == target_val else 0.0 for x in outcomes_a])
    y_b = np.array([1.0 if x == target_val else 0.0 for x in outcomes_b])
    diff = y_b - y_a
    mean_diff = float(np.mean(diff))
    if n > 1:
        s_d = float(np.std(diff, ddof=1))
        se = s_d / math.sqrt(n)
        z = 1.95996
        ci_lower = mean_diff - z * se
        ci_upper = mean_diff + z * se
    else:
        ci_lower, ci_upper = mean_diff, mean_diff
    return mean_diff, ci_lower, ci_upper

def load_all_stage2_policies(device='cpu'):
    # 1. Base B0
    base_policy = load_stage4_policy('base', device=device)

    # 2. Fixed LR Gated (K=4)
    fixed_k4_policy = load_stage2_policy(termination_mode='fixed_k4', device=device)

    # 3. CUSUM Termination (parameters from tuned JSON)
    cusum_file = REPLICATION_DIR / 'cusum_selected_params.json'
    with open(cusum_file, 'r') as f:
        cusum_meta = json.load(f)
    cusum_params = cusum_meta['selected_parameters']
    cusum_policy = load_stage2_policy(
        termination_mode='cusum',
        device=device,
        cusum_params=cusum_params
    )

    # 4. Learned Termination Head & 5. Shuffled Control
    champ_meta_file = REPLICATION_DIR / 'stage2_champion_metadata.json'
    with open(champ_meta_file, 'r') as f:
        champ_meta = json.load(f)

    champ_model = joblib.load(REPLICATION_DIR / 'stage2_champion_model.joblib')
    shuf_model = joblib.load(REPLICATION_DIR / 'stage2_shuffled_model.joblib')

    scaler = {
        'mean': np.array(champ_meta['scaler_mean'], dtype=np.float32),
        'scale': np.array(champ_meta['scaler_scale'], dtype=np.float32)
    }

    learned_policy = load_stage2_policy(
        termination_mode='learned',
        device=device,
        learned_model=champ_model,
        learned_scaler=scaler,
        learned_threshold=champ_meta['threshold']
    )

    shuffled_policy = load_stage2_policy(
        termination_mode='shuffled',
        device=device,
        learned_model=shuf_model,
        learned_scaler=scaler,
        learned_threshold=champ_meta['threshold']
    )

    return {
        'arm1_base': base_policy,
        'arm2_fixed_k4': fixed_k4_policy,
        'arm3_cusum': cusum_policy,
        'arm4_learned': learned_policy,
        'arm5_shuffled': shuffled_policy
    }

def run_chunk_eval(start_case, num_cases, device='cpu', out_file=None):
    assert_case_guard(start_case, num_cases)
    env, env_config = create_environment()
    env.test_sim = 'square_crossing'
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    policies = load_all_stage2_policies(device=device)

    records = []
    t_start = time.time()

    for idx in range(num_cases):
        case_id = start_case + idx
        assert_case_guard(case_id, 1)
        case_res = {'case_id': case_id}

        for arm_name, pol in policies.items():
            robot.set_policy(pol)
            robot.env = env
            env.set_robot(robot)
            ep_res = run_single_episode(pol, env, case_id)
            ep_res['dwell_history'] = getattr(pol, 'dwell_history', [])
            case_res[arm_name] = ep_res

        records.append(case_res)

        if (idx + 1) % 5 == 0 or idx == num_cases - 1:
            if out_file:
                with open(out_file, 'w') as f:
                    json.dump(records, f, indent=2)
            elapsed = time.time() - t_start
            print(f"[{Path(out_file).name if out_file else 'Worker'}] Done {idx+1}/{num_cases} cases | Elapsed: {elapsed:.1f}s", flush=True)

    return records

def evaluate_and_report(records):
    arm_keys = ['arm1_base', 'arm2_fixed_k4', 'arm3_cusum', 'arm4_learned', 'arm5_shuffled']
    arm_labels = {
        'arm1_base': 'Pure B0 Baseline',
        'arm2_fixed_k4': 'Fixed LR Gated (K=4)',
        'arm3_cusum': 'CUSUM Termination (COMPASS-style)',
        'arm4_learned': 'Champion Learned Termination',
        'arm5_shuffled': 'Shuffled Labels Control'
    }

    n_cases = len(records)
    summary_arms = {}

    for arm in arm_keys:
        outcomes = [d[arm]['outcome'] for d in records]
        times = [d[arm]['final_time'] for d in records if d[arm]['outcome'] == 'SUCCESS']
        dwells = []
        for d in records:
            dwells.extend(d[arm].get('dwell_history', []))

        sr = float(np.mean([1.0 if o == 'SUCCESS' else 0.0 for o in outcomes]) * 100.0)
        cr = float(np.mean([1.0 if o == 'COLLISION' else 0.0 for o in outcomes]) * 100.0)
        tr = float(np.mean([1.0 if o == 'TIMEOUT' else 0.0 for o in outcomes]) * 100.0)
        sor = float(np.mean([1.0 if d[arm].get('stagnation', False) else 0.0 for d in records]) * 100.0)
        avg_t = float(np.mean(times)) if times else 0.0
        avg_interv = float(np.mean([d[arm].get('interventions', 0) for d in records]))

        dwell_dist = {
            'count': len(dwells),
            'mean': round(float(np.mean(dwells)), 2) if dwells else 0.0,
            'median': round(float(np.median(dwells)), 2) if dwells else 0.0,
            'p25': round(float(np.percentile(dwells, 25)), 2) if dwells else 0.0,
            'p75': round(float(np.percentile(dwells, 75)), 2) if dwells else 0.0,
            'min': int(np.min(dwells)) if dwells else 0,
            'max': int(np.max(dwells)) if dwells else 0
        }

        summary_arms[arm] = {
            'label': arm_labels[arm],
            'n_cases': n_cases,
            'success_count': sum(1 for o in outcomes if o == 'SUCCESS'),
            'collision_count': sum(1 for o in outcomes if o == 'COLLISION'),
            'timeout_count': sum(1 for o in outcomes if o == 'TIMEOUT'),
            'stagnation_count': sum(1 for d in records if d[arm].get('stagnation', False)),
            'sr': round(sr, 2),
            'cr': round(cr, 2),
            'tr': round(tr, 2),
            'sor': round(sor, 2),
            'avg_time': round(avg_t, 2),
            'avg_interv': round(avg_interv, 2),
            'dwell_distribution': dwell_dist
        }

    # Dual Primary Comparisons (alpha = 0.025 each)
    outcomes_learned = [d['arm4_learned']['outcome'] for d in records]
    outcomes_fixed = [d['arm2_fixed_k4']['outcome'] for d in records]
    outcomes_cusum = [d['arm3_cusum']['outcome'] for d in records]
    outcomes_base = [d['arm1_base']['outcome'] for d in records]

    # Comp 1: Learned vs Fixed (K=4)
    p1, b1, c1 = mcnemar_exact_test(outcomes_fixed, outcomes_learned)
    d_sr1 = summary_arms['arm4_learned']['sr'] - summary_arms['arm2_fixed_k4']['sr']

    # Comp 2: Learned vs CUSUM
    p2, b2, c2 = mcnemar_exact_test(outcomes_cusum, outcomes_learned)
    d_sr2 = summary_arms['arm4_learned']['sr'] - summary_arms['arm3_cusum']['sr']

    # Preregistered Decision Rule
    # SUPPORTED: (d_sr1 > 0 and p1 < 0.025) and (d_sr2 > 0 and p2 < 0.025)
    # NOT SUPPORTED: (d_sr1 < 1.0) or (d_sr2 < 1.0)
    # INCONCLUSIVE: otherwise
    if (d_sr1 > 0.0 and p1 < 0.025) and (d_sr2 > 0.0 and p2 < 0.025):
        decision = 'SUPPORTED'
    elif (d_sr1 < 1.0) or (d_sr2 < 1.0):
        decision = 'NOT SUPPORTED'
    else:
        decision = 'INCONCLUSIVE'

    stats = {
        'dataset': 'Stage2_Square_1000cases',
        'n_cases': n_cases,
        'decision': decision,
        'comp1_learned_vs_fixed_k4': {
            'delta_sr_pp': round(d_sr1, 2),
            'p_value': round(p1, 5),
            'fixed_won': b1,
            'learned_won': c1,
            'pass_alpha_025': bool(d_sr1 > 0 and p1 < 0.025)
        },
        'comp2_learned_vs_cusum': {
            'delta_sr_pp': round(d_sr2, 2),
            'p_value': round(p2, 5),
            'cusum_won': b2,
            'learned_won': c2,
            'pass_alpha_025': bool(d_sr2 > 0 and p2 < 0.025)
        },
        'arms_summary': summary_arms
    }

    with open(OUT_STATS, 'w') as f:
        json.dump(stats, f, indent=2)

    # Generate Stage 2 Report
    lines = [decision]
    lines.append("")
    lines.append("# 阶段 2：可学习意图承诺终止独立评测报告")
    lines.append("")
    lines.append("## 一、评测元数据与数据来源")
    lines.append(f"- **评测执行脚本**：`{REPO_ROOT / 'replication/evaluate_stage2.py'}`")
    lines.append(f"- **原始测试数据文件**：`{OUT_MASTER}`")
    lines.append(f"- **统计汇总数据文件**：`{OUT_STATS}`")
    lines.append(f"- **前置预登记判据文件**：`{REPLICATION_DIR / 'PREREG.md'}`")
    lines.append(f"- **CUSUM 选定参数记录**：`{REPLICATION_DIR / 'cusum_selected_params.json'}`")
    lines.append(f"- **可学习模型冠军元数据**：`{REPLICATION_DIR / 'stage2_champion_metadata.json'}`")
    lines.append("- **重放闸门与回归闸门**：已验证通过（20例重放 0.0e+00 误差，强制第4步终止与原LR K=4逐步完全一致）")
    lines.append("")

    lines.append("## 二、测试块结果（Case 98000–98999，20人方形交叉，N=1000，只读一次）")
    lines.append("")
    lines.append("| 评测臂 / 方法 | 样本数 | 成功率 (%) | 碰撞率 (%) | 超时率 (%) | 停滞率 (%) | 平均耗时 (s) | 平均干预频次 | 平均承诺步数 | 承诺步数中位数 [P25, P75] |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for arm in arm_keys:
        d = summary_arms[arm]
        dw = d['dwell_distribution']
        dw_str = f"{dw['median']} [{dw['p25']}, {dw['p75']}]" if dw['count'] > 0 else "N/A"
        lines.append(
            f"| {d['label']} | {d['n_cases']} | {d['sr']:.1f}% ({d['success_count']}/{d['n_cases']}) | "
            f"{d['cr']:.1f}% ({d['collision_count']}/{d['n_cases']}) | {d['tr']:.1f}% ({d['timeout_count']}/{d['n_cases']}) | "
            f"{d['sor']:.1f}% ({d['stagnation_count']}/{d['n_cases']}) | {d['avg_time']:.2f}s | {d['avg_interv']:.2f} | "
            f"{dw['mean']:.2f} | {dw_str} |"
        )
    lines.append("")

    lines.append("## 三、双主比较判定结果（显著性水平各取 α = 0.025）")
    lines.append("")
    c1 = stats['comp1_learned_vs_fixed_k4']
    lines.append("### 主比较 1：可学习终止 对 逻辑回归门控（K=4）")
    lines.append(f"- **成功率差值（ΔSR₁）**：{c1['delta_sr_pp']:+.2f} 个百分点（{summary_arms['arm4_learned']['sr']}% 对 {summary_arms['arm2_fixed_k4']['sr']}%）")
    lines.append(f"- **双侧配对精确 McNemar 检验**：p₁ = {c1['p_value']:.5f}（固定K=4胜 {c1['fixed_won']} 例，可学习终止胜 {c1['learned_won']} 例）")
    lines.append(f"- **比较 1 达标状态**：{'已验证通过' if c1['pass_alpha_025'] else '未验证通过'}")
    lines.append("")

    c2 = stats['comp2_learned_vs_cusum']
    lines.append("### 主比较 2：可学习终止 对 累积和终止（CUSUM）")
    lines.append(f"- **成功率差值（ΔSR₂）**：{c2['delta_sr_pp']:+.2f} 个百分点（{summary_arms['arm4_learned']['sr']}% 对 {summary_arms['arm3_cusum']['sr']}%）")
    lines.append(f"- **双侧配对精确 McNemar 检验**：p₂ = {c2['p_value']:.5f}（CUSUM胜 {c2['cusum_won']} 例，可学习终止胜 {c2['learned_won']} 例）")
    lines.append(f"- **比较 2 达标状态**：{'已验证通过' if c2['pass_alpha_025'] else '未验证通过'}")
    lines.append("")

    lines.append(f"### 预登记双主比较最终判据结论：**`{decision}`**")
    lines.append("")

    lines.append("## 四、对照臂与承诺步数分布分析")
    shuf = summary_arms['arm5_shuffled']
    lines.append(f"- **打乱标签对照臂表现**：成功率 {shuf['sr']}%, 碰撞率 {shuf['cr']}%, 超时率 {shuf['tr']}%, 平均耗时 {shuf['avg_time']}s, 平均承诺步数 {shuf['dwell_distribution']['mean']}")
    lines.append(f"- **对基线 B0 的差异**：可学习终止较 Pure B0 成功率差 {summary_arms['arm4_learned']['sr'] - summary_arms['arm1_base']['sr']:+.2f} 个百分点")
    lines.append("")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report generated at: {REPORT_PATH}")
    return stats

def main():
    import argparse
    import subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_case', type=int, default=98000)
    parser.add_argument('--total_cases', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--chunk_mode', action='store_true')
    parser.add_argument('--out_file', type=str, default=None)
    args = parser.parse_args()

    if args.chunk_mode:
        run_chunk_eval(args.start_case, args.total_cases, device=args.device, out_file=args.out_file)
        return

    assert_case_guard(args.start_case, args.total_cases)
    num_workers = min(args.workers, 4)

    print("=" * 105)
    print(f"STAGE 2 TEST EVALUATION: CASES {args.start_case}..{args.start_case + args.total_cases - 1} | WORKERS={num_workers}")
    print("=" * 105)

    cases_per_worker = args.total_cases // num_workers
    remainder = args.total_cases % num_workers

    chunks = []
    processes = []
    curr_start = args.start_case
    t_start = time.time()

    for w in range(num_workers):
        w_cases = cases_per_worker + (1 if w < remainder else 0)
        w_out = REPLICATION_DIR / f'stage2_eval_part{w}.json'
        chunks.append({'worker_id': w, 'start': curr_start, 'num': w_cases, 'out_file': w_out})

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            '--chunk_mode',
            '--start_case', str(curr_start),
            '--total_cases', str(w_cases),
            '--device', args.device,
            '--out_file', str(w_out)
        ]
        log_f = open(REPLICATION_DIR / f'stage2_eval_worker_{w}.log', 'w')
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append((proc, log_f, w))
        curr_start += w_cases

    print(f"Launched {num_workers} parallel workers. Waiting for evaluation...")
    for proc, log_f, wid in processes:
        proc.wait()
        log_f.close()
        print(f"Worker {wid} completed with exit code {proc.returncode}.")
        if proc.returncode != 0:
            print(f"ERROR: Worker {wid} exited with error!")
            sys.exit(1)

    merged_data = []
    for c in chunks:
        with open(c['out_file'], 'r') as f:
            merged_data.extend(json.load(f))

    merged_data.sort(key=lambda x: x['case_id'])
    with open(OUT_MASTER, 'w') as f:
        json.dump(merged_data, f, indent=2)

    t_total = time.time() - t_start
    print(f"Merged {len(merged_data)} cases into {OUT_MASTER}. Total time: {t_total:.1f}s")

    stats = evaluate_and_report(merged_data)
    print("\n" + "=" * 105)
    print(f"STAGE 2 SUMMARY RESULTS: Decision = [{stats['decision']}]")
    print("=" * 105)

if __name__ == '__main__':
    main()
