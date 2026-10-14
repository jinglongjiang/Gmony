#!/usr/bin/env python3
"""Regression Gate Check: Verify Stage 4 Parity on Cases 90300 - 90309.

Runs the four frozen arms:
- Arm 1: Base Single-Stream GRU (B0)
- Arm 2: Heuristic Gated (K=4)
- Arm 3: Champion Learned Gate - Random Forest (K=4)
- Arm 4: Champion Learned Gate - Logistic Regression (K=4)

Compares per-episode outcomes against benchmark_results/stage4_500cases_final_evaluation.json.
Must match 100% (10/10 cases, 40/40 runs) or halt immediately.
"""

import sys
import os
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KDARL_CLEAN = REPO_ROOT / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(REPO_ROOT)] + [p for p in sys.path if p not in (str(REPO_ROOT), str(KDARL_CLEAN), '')]

from crowd_sim.envs.utils.robot import Robot
from run_b0_counterfactual_study import create_environment
from run_stage4_500cases_evaluation import load_stage4_policy, run_single_episode

def run_regression_gate(start_case=90300, num_cases=10, device='cuda'):
    gold_path = REPO_ROOT / 'benchmark_results/stage4_500cases_final_evaluation.json'
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold benchmark data not found at {gold_path}")

    with open(gold_path, 'r') as f:
        gold_data = json.load(f)

    gold_map = {d['case_id']: d for d in gold_data if start_case <= d['case_id'] < start_case + num_cases}
    print(f"Loaded {len(gold_map)} reference cases from {gold_path.name}")

    env, env_config = create_environment()
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    policies = {
        'arm1_base': load_stage4_policy('base', device=device),
        'arm2_heuristic': load_stage4_policy('heuristic', device=device),
        'arm3_learned_rf': load_stage4_policy('learned_rf', device=device),
        'arm4_learned_lr': load_stage4_policy('learned_lr', device=device)
    }

    arm_keys = ['arm1_base', 'arm2_heuristic', 'arm3_learned_rf', 'arm4_learned_lr']
    all_matched = True
    mismatches = []
    run_results = []

    print("=" * 90)
    print(f"RUNNING REGRESSION GATE ON CASES {start_case}..{start_case + num_cases - 1}")
    print("=" * 90)

    for case_id in range(start_case, start_case + num_cases):
        if case_id not in gold_map:
            raise KeyError(f"Case {case_id} not present in reference gold data!")

        ref_case = gold_map[case_id]
        case_record = {'case_id': case_id}

        for arm_name in arm_keys:
            pol = policies[arm_name]
            robot.set_policy(pol)
            robot.env = env
            env.set_robot(robot)

            ep_res = run_single_episode(pol, env, case_id)
            case_record[arm_name] = ep_res

            ref_outcome = ref_case[arm_name]['outcome']
            act_outcome = ep_res['outcome']

            ref_time = ref_case[arm_name]['final_time']
            act_time = ep_res['final_time']

            matched = (ref_outcome == act_outcome)
            status_str = "PASS" if matched else "FAIL"
            print(f"Case {case_id} | {arm_name:<16} | Ref: {ref_outcome:<9} ({ref_time:5.2f}s) | Act: {act_outcome:<9} ({act_time:5.2f}s) | [{status_str}]")

            if not matched:
                all_matched = False
                mismatches.append({
                    'case_id': case_id,
                    'arm': arm_name,
                    'ref_outcome': ref_outcome,
                    'act_outcome': act_outcome,
                    'ref_time': ref_time,
                    'act_time': act_time
                })

        run_results.append(case_record)

    out_json = REPO_ROOT / 'replication/regression_gate_results.json'
    with open(out_json, 'w') as f:
        json.dump({
            'status': 'PASS' if all_matched else 'FAIL',
            'start_case': start_case,
            'num_cases': num_cases,
            'all_matched': all_matched,
            'mismatch_count': len(mismatches),
            'mismatches': mismatches,
            'results': run_results
        }, f, indent=2)

    print("=" * 90)
    if all_matched:
        print(f"REGRESSION GATE RESULT: ALL {num_cases * 4}/{num_cases * 4} RUNS MATCHED EXACTLY (100% PARITY).")
        print(f"Gate output saved to: {out_json}")
    else:
        print(f"REGRESSION GATE FAILED: {len(mismatches)} mismatches detected!")
        for m in mismatches:
            print(f"  - Case {m['case_id']}, {m['arm']}: Ref={m['ref_outcome']}, Act={m['act_outcome']}")
    print("=" * 90)

    return all_matched

if __name__ == '__main__':
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    success = run_regression_gate(start_case=90300, num_cases=10, device=dev)
    if not success:
        sys.exit(1)
