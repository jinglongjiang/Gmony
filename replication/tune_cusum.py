import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import sys
import json
import math
import time
import subprocess
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)

REPO_ROOT = Path('/home/abc/workspace/autonomous_research_20261012')
sys.path.insert(0, str(REPO_ROOT))

from crowd_sim.envs.utils.robot import Robot
from run_b0_counterfactual_study import create_environment
from run_stage4_500cases_evaluation import run_single_episode
from replication.stage2_policy import load_stage2_policy

REPLICATION_DIR = REPO_ROOT / 'replication'
OUT_SELECTED_PARAMS = REPLICATION_DIR / 'cusum_selected_params.json'
OUT_GRID_RESULTS = REPLICATION_DIR / 'cusum_grid_search_results.json'

FORBIDDEN_RANGES = [
    (92000, 92499),
    (90000, 90299),
    (93000, 93199),
    (95000, 96499),
    (98000, 98999)
]

def assert_case_guard(start_case, num_cases):
    end_case = start_case + num_cases - 1
    for f_start, f_end in FORBIDDEN_RANGES:
        if not (end_case < f_start or start_case > f_end):
            raise ValueError(
                f"SAFETY VIOLATION: Case range [{start_case}, {end_case}] overlaps "
                f"with strictly prohibited range [{f_start}, {f_end}]!"
            )

# Pre-defined candidate grid for CUSUM parameters
# Parameters: decay (lambda), threshold (E_th), floor (Delta_floor)
CANDIDATE_CUSUM_GRID = [
    {'name': 'cusum_cand1_fast', 'decay': 0.80, 'threshold': 0.10, 'floor': 0.00},
    {'name': 'cusum_cand2_med_low', 'decay': 0.85, 'threshold': 0.15, 'floor': 0.00},
    {'name': 'cusum_cand3_med', 'decay': 0.90, 'threshold': 0.20, 'floor': 0.00},
    {'name': 'cusum_cand4_med_floor', 'decay': 0.90, 'threshold': 0.15, 'floor': 0.05},
    {'name': 'cusum_cand5_slow', 'decay': 0.95, 'threshold': 0.25, 'floor': 0.00},
    {'name': 'cusum_cand6_slow_floor', 'decay': 0.95, 'threshold': 0.20, 'floor': 0.05},
    {'name': 'cusum_cand7_conservative', 'decay': 0.98, 'threshold': 0.35, 'floor': 0.05},
    {'name': 'cusum_cand8_persistent', 'decay': 1.00, 'threshold': 0.30, 'floor': 0.10},
]

def evaluate_param_set(param_set, val_cases, device='cpu', print_case_progress=True):
    env, env_config = create_environment()
    env.test_sim = 'square_crossing'
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    policy = load_stage2_policy(
        termination_mode='cusum',
        device=device,
        cusum_params=param_set
    )
    robot.set_policy(policy)
    robot.env = env
    env.set_robot(robot)

    n_cases = len(val_cases)
    outcomes = []
    times = []
    dwell_all = []
    cand_name = param_set.get('name', 'custom')

    t_eval_start = time.time()
    for idx, case_id in enumerate(val_cases):
        assert_case_guard(case_id, 1)
        t_case = time.time()
        res = run_single_episode(policy, env, case_id)
        dt_case = time.time() - t_case
        outcomes.append(res['outcome'])
        if res['outcome'] == 'SUCCESS':
            times.append(res['final_time'])
        dwell_all.extend(policy.dwell_history)

        if print_case_progress:
            cum_sr = float(np.mean([1.0 if o == 'SUCCESS' else 0.0 for o in outcomes]) * 100.0)
            print(
                f"[{cand_name}] Case {case_id} ({idx+1}/{n_cases}) in {dt_case:.1f}s | "
                f"outcome={res['outcome']} | cum_SR={cum_sr:.1f}% | "
                f"last_dwell={policy.dwell_history[-1] if policy.dwell_history else 0} | "
                f"elapsed={time.time() - t_eval_start:.1f}s",
                flush=True
            )

    sr = float(np.mean([1.0 if o == 'SUCCESS' else 0.0 for o in outcomes]) * 100.0)
    cr = float(np.mean([1.0 if o == 'COLLISION' else 0.0 for o in outcomes]) * 100.0)
    tr = float(np.mean([1.0 if o == 'TIMEOUT' else 0.0 for o in outcomes]) * 100.0)
    avg_t = float(np.mean(times)) if times else 0.0
    mean_dwell = float(np.mean(dwell_all)) if dwell_all else 0.0

    return {
        'name': cand_name,
        'decay': param_set['decay'],
        'threshold': param_set['threshold'],
        'floor': param_set['floor'],
        'n_val_cases': n_cases,
        'success_rate': round(sr, 2),
        'collision_rate': round(cr, 2),
        'timeout_rate': round(tr, 2),
        'avg_time': round(avg_t, 2),
        'mean_dwell_steps': round(mean_dwell, 2),
        'dwell_count': len(dwell_all)
    }

def run_chunk_candidates(cand_indices, out_file, device='cpu'):
    val_cases = list(range(99280, 99400))
    assert_case_guard(val_cases[0], len(val_cases))

    results = []
    for c_idx in cand_indices:
        cand = CANDIDATE_CUSUM_GRID[c_idx]
        t0 = time.time()
        print(f"--- Starting {cand['name']} (idx={c_idx}) ---", flush=True)
        res = evaluate_param_set(cand, val_cases, device=device, print_case_progress=True)
        elapsed = time.time() - t0
        res['eval_time_s'] = round(elapsed, 1)
        results.append(res)
        print(f"[{res['name']:<22}] FINISHED | decay={res['decay']:.2f}, thr={res['threshold']:.2f}, floor={res['floor']:.2f} | "
              f"SR={res['success_rate']:5.1f}% | CR={res['collision_rate']:4.1f}% | Time={res['avg_time']:5.2f}s | "
              f"MeanDwell={res['mean_dwell_steps']:4.1f} | Elapsed={elapsed:.1f}s", flush=True)

        if out_file:
            with open(out_file, 'w') as f:
                json.dump(results, f, indent=2)

    return results

def run_tuning(device='cpu', workers=4):
    val_cases = list(range(99280, 99400))
    assert_case_guard(val_cases[0], len(val_cases))

    print("=" * 95)
    print(f"CUSUM PARAMETER TUNING ON VALIDATION SPLIT (Cases {val_cases[0]}..{val_cases[-1]}, N={len(val_cases)})")
    print(f"Candidates to evaluate: {len(CANDIDATE_CUSUM_GRID)} | Workers: {workers}")
    print("=" * 95, flush=True)

    t_start = time.time()
    results = []

    if workers <= 1:
        for idx, cand in enumerate(CANDIDATE_CUSUM_GRID):
            t0 = time.time()
            res = evaluate_param_set(cand, val_cases, device=device, print_case_progress=True)
            elapsed = time.time() - t0
            res['eval_time_s'] = round(elapsed, 1)
            results.append(res)
            print(f"[{res['name']:<22}] decay={res['decay']:.2f}, thr={res['threshold']:.2f}, floor={res['floor']:.2f} | "
                  f"SR={res['success_rate']:5.1f}% | CR={res['collision_rate']:4.1f}% | Time={res['avg_time']:5.2f}s | "
                  f"MeanDwell={res['mean_dwell_steps']:4.1f} | Elapsed={elapsed:.1f}s", flush=True)
            with open(OUT_GRID_RESULTS, 'w') as f:
                json.dump(results, f, indent=2)
    else:
        num_workers = min(workers, len(CANDIDATE_CUSUM_GRID))
        worker_cand_lists = [[] for _ in range(num_workers)]
        for i in range(len(CANDIDATE_CUSUM_GRID)):
            worker_cand_lists[i % num_workers].append(i)

        processes = []
        part_files = []
        for w in range(num_workers):
            part_out = REPLICATION_DIR / f'cusum_tune_part{w}.json'
            part_files.append(part_out)
            indices_str = ','.join(str(idx) for idx in worker_cand_lists[w])
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                '--chunk_mode',
                '--cand_indices', indices_str,
                '--out_file', str(part_out),
                '--device', device
            ]
            log_f = open(REPLICATION_DIR / f'cusum_tune_worker_{w}.log', 'w')
            sub_env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=sub_env)
            processes.append((proc, log_f, w, worker_cand_lists[w]))
            print(f"Launched CUSUM Worker {w} for candidates {worker_cand_lists[w]} -> log: cusum_tune_worker_{w}.log", flush=True)

        for proc, log_f, wid, c_list in processes:
            proc.wait()
            log_f.close()
            print(f"CUSUM Worker {wid} (candidates {c_list}) finished with exit code {proc.returncode}.", flush=True)
            if proc.returncode != 0:
                raise RuntimeError(f"CUSUM Worker {wid} failed! Check cusum_tune_worker_{wid}.log")

        for pf in part_files:
            with open(pf, 'r') as f:
                results.extend(json.load(f))

    # Select champion: Highest SR -> Lowest CR -> Lowest avg_time
    results.sort(key=lambda x: (-x['success_rate'], x['collision_rate'], x['avg_time']))
    champion = results[0]

    with open(OUT_GRID_RESULTS, 'w') as f:
        json.dump(results, f, indent=2)

    selected = {
        'reference': 'COMPASS (Kang, 2026 pre-print)',
        'tuning_dataset': f"Cases {val_cases[0]}..{val_cases[-1]} (Val Split N={len(val_cases)})",
        'selected_parameters': {
            'decay': champion['decay'],
            'threshold': champion['threshold'],
            'floor': champion['floor']
        },
        'champion_name': champion['name'],
        'validation_metrics': {
            'success_rate': champion['success_rate'],
            'collision_rate': champion['collision_rate'],
            'timeout_rate': champion['timeout_rate'],
            'avg_time': champion['avg_time'],
            'mean_dwell_steps': champion['mean_dwell_steps']
        },
        'all_evaluated_candidates': results
    }

    with open(OUT_SELECTED_PARAMS, 'w') as f:
        json.dump(selected, f, indent=2)

    total_elapsed = time.time() - t_start
    print("=" * 95)
    print(f"CUSUM SELECTION WINNER: {champion['name']}")
    print(f"  decay={champion['decay']}, threshold={champion['threshold']}, floor={champion['floor']}")
    print(f"  Validation SR={champion['success_rate']}%, CR={champion['collision_rate']}%, Time={champion['avg_time']}s")
    print(f"Saved selected parameters to: {OUT_SELECTED_PARAMS}")
    print(f"Saved all grid results to: {OUT_GRID_RESULTS}")
    print(f"Total Tuning Wall Time: {total_elapsed:.1f}s")
    print("=" * 95, flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--chunk_mode', action='store_true')
    parser.add_argument('--cand_indices', type=str, default='')
    parser.add_argument('--out_file', type=str, default=None)
    args = parser.parse_args()

    if args.chunk_mode:
        c_indices = [int(x.strip()) for x in args.cand_indices.split(',') if x.strip()]
        run_chunk_candidates(c_indices, args.out_file, device=args.device)
    else:
        run_tuning(device=args.device, workers=args.workers)
