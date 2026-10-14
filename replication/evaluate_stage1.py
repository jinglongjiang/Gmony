#!/usr/bin/env python3
"""Stage 1 Evaluation Runner: Independent Replication Benchmark.

Evaluates 4 frozen arms under identical CrowdSim contracts:
- Arm 1: Base Single-Stream GRU (B0)
- Arm 2: Heuristic Gated (K=4)
- Arm 3: Champion Learned Gate - Random Forest (K=4)
- Arm 4: Champion Learned Gate - Logistic Regression (K=4)

Phases:
- smoke: Cases 97000 - 97009 (10 cases, square crossing)
- square: Cases 97000 - 97999 (1000 cases, square crossing, primary preregistered comparison)
- circle: Cases 97000 - 97499 (500 cases, circle crossing, secondary generalization observation)

Enforces strict case boundary guards:
- Raises error if any case is in:
  [92000..92499, 90000..90299, 93000..93199, 95000..96499]
"""

import sys
import os
import math
import json
import time
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
from scipy.stats import binom_test

REPO_ROOT = Path(__file__).resolve().parent.parent
KDARL_CLEAN = REPO_ROOT / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(REPO_ROOT)] + [p for p in sys.path if p not in (str(REPO_ROOT), str(KDARL_CLEAN), '')]

from crowd_sim.envs.utils.robot import Robot
from run_b0_counterfactual_study import create_environment
from run_stage4_500cases_evaluation import load_stage4_policy, run_single_episode

REPLICATION_DIR = REPO_ROOT / 'replication'
REPLICATION_DIR.mkdir(parents=True, exist_ok=True)

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

def run_chunk_benchmark(start_case, num_cases, sim_rule='square_crossing', device='cuda', out_file=None):
    assert_case_guard(start_case, num_cases)
    env, env_config = create_environment()
    env.test_sim = sim_rule
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    policies = {
        'arm1_base': load_stage4_policy('base', device=device),
        'arm2_heuristic': load_stage4_policy('heuristic', device=device),
        'arm3_learned_rf': load_stage4_policy('learned_rf', device=device),
        'arm4_learned_lr': load_stage4_policy('learned_lr', device=device)
    }

    records = []
    t_start = time.time()

    for idx in range(num_cases):
        case_id = start_case + idx
        assert_case_guard(case_id, 1)
        case_res = {'case_id': case_id, 'sim_rule': sim_rule}

        for arm_name, pol in policies.items():
            robot.set_policy(pol)
            robot.env = env
            env.set_robot(robot)
            ep_res = run_single_episode(pol, env, case_id)
            case_res[arm_name] = ep_res

        records.append(case_res)

        if (idx + 1) % 5 == 0 or idx == num_cases - 1:
            if out_file:
                with open(out_file, 'w') as f:
                    json.dump(records, f, indent=2)
                elapsed = time.time() - t_start
                print(f"[{Path(out_file).name}] Done {idx+1}/{num_cases} cases | Elapsed: {elapsed:.1f}s ({elapsed/(idx+1):.2f}s/case)", flush=True)

    return records

def mcnemar_exact_test(arm_a_outcomes, arm_b_outcomes):
    """Computes exact two-sided McNemar test on discordant pairs."""
    b = sum(1 for a, b in zip(arm_a_outcomes, arm_b_outcomes) if a == 'SUCCESS' and b != 'SUCCESS')
    c = sum(1 for a, b in zip(arm_a_outcomes, arm_b_outcomes) if a != 'SUCCESS' and b == 'SUCCESS')
    n = b + c
    if n == 0:
        return 1.0, b, c
    pval = binom_test(min(b, c), n, 0.5)
    return float(pval), b, c

def compute_paired_ci_diff(outcomes_a, outcomes_b, target_val='COLLISION'):
    """Computes sample paired difference and 95% Wald/score confidence interval for target_val."""
    n = len(outcomes_a)
    y_a = np.array([1.0 if x == target_val else 0.0 for x in outcomes_a])
    y_b = np.array([1.0 if x == target_val else 0.0 for x in outcomes_b])
    diff = y_b - y_a  # difference (B - A)
    mean_diff = float(np.mean(diff))
    if n > 1:
        s_d = float(np.std(diff, ddof=1))
        se = s_d / math.sqrt(n)
        z = 1.95996
        ci_lower = mean_diff - z * se
        ci_upper = mean_diff + z * se
    else:
        ci_lower = mean_diff
        ci_upper = mean_diff
    return mean_diff, ci_lower, ci_upper

def evaluate_metrics(records, dataset_name="Stage1_Square_1000"):
    arm_keys = ['arm1_base', 'arm2_heuristic', 'arm3_learned_rf', 'arm4_learned_lr']
    arm_labels = {
        'arm1_base': 'Pure B0 Baseline',
        'arm2_heuristic': 'Heuristic Gated (K=4)',
        'arm3_learned_rf': 'Champion Learned RF (K=4)',
        'arm4_learned_lr': 'Champion Learned LR (K=4)'
    }

    n_cases = len(records)
    stats = {}
    for arm in arm_keys:
        outcomes = [d[arm]['outcome'] for d in records]
        times = [d[arm]['final_time'] for d in records if d[arm]['outcome'] == 'SUCCESS']
        sr = float(np.mean([1.0 if o == 'SUCCESS' else 0.0 for o in outcomes]) * 100.0)
        cr = float(np.mean([1.0 if o == 'COLLISION' else 0.0 for o in outcomes]) * 100.0)
        tr = float(np.mean([1.0 if o == 'TIMEOUT' else 0.0 for o in outcomes]) * 100.0)
        sor = float(np.mean([1.0 if d[arm].get('stagnation', False) else 0.0 for d in records]) * 100.0)
        avg_t = float(np.mean(times)) if times else 0.0
        avg_interv = float(np.mean([d[arm].get('interventions', 0) for d in records]))

        stats[arm] = {
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
            'outcomes': outcomes
        }

    # Primary Comparison: Arm 4 (Learned LR) vs Arm 1 (Base B0)
    base_outs = stats['arm1_base']['outcomes']
    lr_outs = stats['arm4_learned_lr']['outcomes']

    p_val_lr, b_lr, c_lr = mcnemar_exact_test(base_outs, lr_outs)
    delta_sr = (stats['arm4_learned_lr']['sr'] - stats['arm1_base']['sr'])
    cr_diff_mean, cr_ci_low, cr_ci_up = compute_paired_ci_diff(base_outs, lr_outs, 'COLLISION')

    delta_cr_pct = cr_diff_mean * 100.0
    cr_ci_low_pct = cr_ci_low * 100.0
    cr_ci_up_pct = cr_ci_up * 100.0

    # Preregistered Decision Rules:
    # REPLICATED: delta_sr > 0 and p_val < 0.05 and cr_ci_up_pct < +2.0
    # NOT REPLICATED: delta_sr < +1.0
    # INCONCLUSIVE: otherwise
    if (delta_sr > 0.0) and (p_val_lr < 0.05) and (cr_ci_up_pct < 2.0):
        decision = 'REPLICATED'
    elif delta_sr < 1.0:
        decision = 'NOT REPLICATED'
    else:
        decision = 'INCONCLUSIVE'

    # Secondary comparisons for report
    p_val_heu, b_heu, c_heu = mcnemar_exact_test(base_outs, stats['arm2_heuristic']['outcomes'])
    p_val_rf, b_rf, c_rf = mcnemar_exact_test(base_outs, stats['arm3_learned_rf']['outcomes'])

    summary = {
        'dataset': dataset_name,
        'n_cases': n_cases,
        'decision': decision,
        'primary_comparison': {
            'comparison': 'Arm 4 (Learned LR Gated) vs Arm 1 (Base B0)',
            'delta_sr_pp': round(delta_sr, 2),
            'mcnemar_p_value': round(p_val_lr, 5),
            'base_won_b': b_lr,
            'candidate_won_c': c_lr,
            'delta_cr_pp': round(delta_cr_pct, 2),
            'cr_diff_95ci_low_pp': round(cr_ci_low_pct, 2),
            'cr_diff_95ci_up_pp': round(cr_ci_up_pct, 2),
            'cr_bound_pass': bool(cr_ci_up_pct < 2.0)
        },
        'secondary_comparisons': {
            'heuristic_vs_base': {
                'delta_sr_pp': round(stats['arm2_heuristic']['sr'] - stats['arm1_base']['sr'], 2),
                'p_value': round(p_val_heu, 5),
                'b': b_heu,
                'c': c_heu
            },
            'rf_vs_base': {
                'delta_sr_pp': round(stats['arm3_learned_rf']['sr'] - stats['arm1_base']['sr'], 2),
                'p_value': round(p_val_rf, 5),
                'b': b_rf,
                'c': c_rf
            }
        },
        'arms_summary': {arm: {k: v for k, v in stats[arm].items() if k != 'outcomes'} for arm in arm_keys}
    }

    return summary

def main():
    parser = argparse.ArgumentParser(description="Stage 1 Replication Evaluation Runner")
    parser.add_argument('--phase', type=str, choices=['smoke', 'square', 'circle'], required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--chunk_mode', action='store_true')
    parser.add_argument('--start_case', type=int, default=None)
    parser.add_argument('--num_cases', type=int, default=None)
    parser.add_argument('--sim_rule', type=str, default='square_crossing')
    parser.add_argument('--out_file', type=str, default=None)
    args = parser.parse_args()

    if args.chunk_mode:
        run_chunk_benchmark(
            args.start_case,
            args.num_cases,
            sim_rule=args.sim_rule,
            device=args.device,
            out_file=args.out_file
        )
        return

    # Phase configuration
    if args.phase == 'smoke':
        start_case = 97000
        total_cases = 10
        sim_rule = 'square_crossing'
        dataset_name = "Smoke_10cases_Square"
        out_master = REPLICATION_DIR / 'smoke_10cases_results.json'
        stats_out = REPLICATION_DIR / 'smoke_10cases_stats.json'
        num_workers = 1
    elif args.phase == 'square':
        start_case = 97000
        total_cases = 1000
        sim_rule = 'square_crossing'
        dataset_name = "Stage1_Square_1000cases"
        out_master = REPLICATION_DIR / 'stage1_square_1000cases.json'
        stats_out = REPLICATION_DIR / 'stage1_square_stats.json'
        num_workers = min(args.workers, 4)
    elif args.phase == 'circle':
        start_case = 97000
        total_cases = 500
        sim_rule = 'circle_crossing'
        dataset_name = "Stage1_Circle_500cases"
        out_master = REPLICATION_DIR / 'stage1_circle_500cases.json'
        stats_out = REPLICATION_DIR / 'stage1_circle_stats.json'
        num_workers = min(args.workers, 4)

    assert_case_guard(start_case, total_cases)

    print("=" * 105)
    print(f"STAGE 1 BENCHMARK: PHASE={args.phase.upper()} | CASES {start_case}..{start_case + total_cases - 1} | SIM={sim_rule} | WORKERS={num_workers}")
    print("=" * 105)

    cases_per_worker = total_cases // num_workers
    remainder = total_cases % num_workers

    chunks = []
    processes = []
    curr_start = start_case
    t_start = time.time()

    for w in range(num_workers):
        w_cases = cases_per_worker + (1 if w < remainder else 0)
        w_out = REPLICATION_DIR / f'stage1_{args.phase}_part{w}.json'
        chunks.append({'worker_id': w, 'start': curr_start, 'num': w_cases, 'out_file': w_out})

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            '--phase', args.phase,
            '--chunk_mode',
            '--start_case', str(curr_start),
            '--num_cases', str(w_cases),
            '--sim_rule', sim_rule,
            '--device', args.device,
            '--out_file', str(w_out)
        ]
        log_f = open(REPLICATION_DIR / f'stage1_{args.phase}_worker_{w}.log', 'w')
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append((proc, log_f, w))
        curr_start += w_cases

    print(f"Launched {num_workers} parallel workers. Waiting for execution to complete...")
    for proc, log_f, wid in processes:
        proc.wait()
        log_f.close()
        print(f"Worker {wid} completed with exit code {proc.returncode}.")
        if proc.returncode != 0:
            print(f"ERROR: Worker {wid} exited with non-zero status! Inspect log: {REPLICATION_DIR / f'stage1_{args.phase}_worker_{wid}.log'}")
            sys.exit(1)

    # Merge results
    merged_data = []
    for c in chunks:
        if c['out_file'].exists():
            with open(c['out_file'], 'r') as f:
                merged_data.extend(json.load(f))
        else:
            raise FileNotFoundError(f"Missing output file: {c['out_file']}")

    merged_data.sort(key=lambda x: x['case_id'])
    with open(out_master, 'w') as f:
        json.dump(merged_data, f, indent=2)

    t_total = time.time() - t_start
    print(f"All {len(merged_data)} cases merged into {out_master}. Total time: {t_total:.1f}s ({t_total/60:.2f} mins)")

    # Compute metrics and preregistered decision
    summary = evaluate_metrics(merged_data, dataset_name=dataset_name)
    with open(stats_out, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 105)
    print(f"SUMMARY RESULTS: {dataset_name} (N={len(merged_data)})")
    print("=" * 105)
    print(f"{'Method / Arm':<28} | {'Success%':<9} | {'Collision%':<11} | {'Timeout%':<9} | {'Stagnation%':<12} | {'Time (s)':<9} | {'Interv/Ep'}")
    print("-" * 105)
    for arm, d in summary['arms_summary'].items():
        print(f"{d['label']:<28} | {d['sr']:7.1f}% | {d['cr']:9.1f}% | {d['tr']:7.1f}% | {d['sor']:10.1f}% | {d['avg_time']:7.2f}s  | {d['avg_interv']:6.1f}")
    print("-" * 105)

    pc = summary['primary_comparison']
    print(f"Primary Comparison: {pc['comparison']}")
    print(f"  Delta SR: {pc['delta_sr_pp']:+.2f} pp")
    print(f"  McNemar p-value: {pc['mcnemar_p_value']:.5f} (Base won {pc['base_won_b']}, LR won {pc['candidate_won_c']})")
    print(f"  Delta CR: {pc['delta_cr_pp']:+.2f} pp (95% CI: [{pc['cr_diff_95ci_low_pp']:+.2f}, {pc['cr_diff_95ci_up_pp']:+.2f}] pp)")
    print(f"  Preregistered Replication Decision: [{summary['decision']}]")
    print("=" * 105)

if __name__ == '__main__':
    main()
