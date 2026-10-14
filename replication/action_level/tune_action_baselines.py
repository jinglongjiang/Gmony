import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import sys
import json
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
from replication.action_level.action_level_policy import load_action_level_policy

ACTION_LEVEL_DIR = REPO_ROOT / 'replication/action_level'
PARAM_LOCK_FILE = ACTION_LEVEL_DIR / 'param_lock.json'
TUNE_RESULTS_FILE = ACTION_LEVEL_DIR / 'tune_results.json'

DEV_START_CASE = 310000
DEV_NUM_CASES = 200

# Strict case guards for Dev Block
FORBIDDEN_RANGES = [
    (97000, 97999),
    (98000, 98999),
    (99000, 99999),
    (90000, 90299),
    (92000, 92499),
    (93000, 93199),
    (95000, 96499)
]

def assert_dev_guard(case_id):
    if not (DEV_START_CASE <= case_id < DEV_START_CASE + DEV_NUM_CASES):
        raise ValueError(f"Case {case_id} is outside the allowed Dev Block [{DEV_START_CASE}, {DEV_START_CASE + DEV_NUM_CASES - 1}]!")
    for f_start, f_end in FORBIDDEN_RANGES:
        if f_start <= case_id <= f_end:
            raise ValueError(f"SAFETY VIOLATION: Case {case_id} falls into forbidden range [{f_start}, {f_end}]!")

CANDIDATES = {
    'hyst_alpha_0.05': {'type': 'hysteresis', 'mode': 'action_hysteresis', 'alpha_a': 0.05, 'k_repeat': 1},
    'hyst_alpha_0.10': {'type': 'hysteresis', 'mode': 'action_hysteresis', 'alpha_a': 0.10, 'k_repeat': 1},
    'hyst_alpha_0.20': {'type': 'hysteresis', 'mode': 'action_hysteresis', 'alpha_a': 0.20, 'k_repeat': 1},
    'repeat_k_2': {'type': 'repeat', 'mode': 'action_repeat', 'alpha_a': 0.0, 'k_repeat': 2},
    'repeat_k_3': {'type': 'repeat', 'mode': 'action_repeat', 'alpha_a': 0.0, 'k_repeat': 3},
    'repeat_k_5': {'type': 'repeat', 'mode': 'action_repeat', 'alpha_a': 0.0, 'k_repeat': 5},
}

def evaluate_single_candidate(cand_name, out_file=None, device='cpu'):
    if cand_name not in CANDIDATES:
        raise ValueError(f"Unknown candidate {cand_name}")

    spec = CANDIDATES[cand_name]
    env, env_cfg = create_environment()
    env.test_sim = 'square_crossing'
    env.human_num = 20
    robot = Robot(env_cfg, 'robot')
    env.set_robot(robot)

    policy = load_action_level_policy(
        mode=spec['mode'],
        alpha_a=spec['alpha_a'],
        k_repeat=spec['k_repeat'],
        device=device
    )
    robot.set_policy(policy)
    robot.env = env
    env.set_robot(robot)

    cases = list(range(DEV_START_CASE, DEV_START_CASE + DEV_NUM_CASES))
    episodes = []
    t_start = time.time()

    for idx, case_id in enumerate(cases):
        assert_dev_guard(case_id)
        t_case0 = time.time()
        res = run_single_episode(policy, env, case_id)
        sw = policy.get_switching_rates()
        dt_case = time.time() - t_case0

        record = {
            'case_id': case_id,
            'outcome': res['outcome'],
            'final_time': res['final_time'],
            'total_steps': res['total_steps'],
            'min_clearance': res['min_clearance'],
            'grid_switch_count': sw['grid_switch_count'],
            'grid_switch_rate': sw['grid_switch_rate'],
            'vel_switch_rate': sw['vel_switch_rate']
        }
        episodes.append(record)

        cum_sr = float(np.mean([1.0 if r['outcome'] == 'SUCCESS' else 0.0 for r in episodes]) * 100.0)
        cum_cr = float(np.mean([1.0 if r['outcome'] == 'COLLISION' else 0.0 for r in episodes]) * 100.0)
        cum_sw = float(np.mean([r['grid_switch_rate'] for r in episodes]))

        if (idx + 1) % 10 == 0 or idx == len(cases) - 1:
            print(
                f"[{cand_name}] Case {case_id} ({idx+1}/{len(cases)}) in {dt_case:.1f}s | "
                f"cum_SR={cum_sr:.1f}% | cum_CR={cum_cr:.1f}% | cum_SwRate={cum_sw:.3f} | "
                f"elapsed={time.time() - t_start:.1f}s",
                flush=True
            )

        if out_file and ((idx + 1) % 25 == 0 or idx == len(cases) - 1):
            with open(out_file, 'w') as f:
                json.dump(episodes, f, indent=2)

    sr = float(np.mean([1.0 if r['outcome'] == 'SUCCESS' else 0.0 for r in episodes]) * 100.0)
    cr = float(np.mean([1.0 if r['outcome'] == 'COLLISION' else 0.0 for r in episodes]) * 100.0)
    tr = float(np.mean([1.0 if r['outcome'] == 'TIMEOUT' else 0.0 for r in episodes]) * 100.0)
    times = [r['final_time'] for r in episodes if r['outcome'] == 'SUCCESS']
    avg_t = float(np.mean(times)) if times else 0.0
    avg_grid_sw = float(np.mean([r['grid_switch_rate'] for r in episodes]))
    avg_vel_sw = float(np.mean([r['vel_switch_rate'] for r in episodes]))
    total_time = round(time.time() - t_start, 1)

    summary = {
        'name': cand_name,
        'type': spec['type'],
        'mode': spec['mode'],
        'alpha_a': spec['alpha_a'],
        'k_repeat': spec['k_repeat'],
        'n_cases': len(cases),
        'success_rate': round(sr, 2),
        'collision_rate': round(cr, 2),
        'timeout_rate': round(tr, 2),
        'avg_time': round(avg_t, 2),
        'mean_grid_switch_rate': round(avg_grid_sw, 4),
        'mean_vel_switch_rate': round(avg_vel_sw, 4),
        'wall_time_s': total_time
    }
    return summary

def run_tuning(device='cpu', workers=6):
    print("=" * 95)
    print(f"DEVELOPMENT BLOCK TUNING: 20-Human Square Crossing (Cases {DEV_START_CASE}..{DEV_START_CASE + DEV_NUM_CASES - 1}, N={DEV_NUM_CASES})")
    print(f"Evaluating {len(CANDIDATES)} candidates: alpha_a in [0.05, 0.10, 0.20], k in [2, 3, 5] | Workers: {workers}")
    print("=" * 95, flush=True)

    t_start = time.time()
    cand_names = list(CANDIDATES.keys())
    results = {}

    if workers <= 1:
        for name in cand_names:
            res = evaluate_single_candidate(name, ACTION_LEVEL_DIR / f'tune_episodes_{name}.json', device=device)
            results[name] = res
    else:
        num_workers = min(workers, len(cand_names))
        processes = []
        part_summaries = []

        for name in cand_names:
            out_episodes = ACTION_LEVEL_DIR / f'tune_episodes_{name}.json'
            out_summary = ACTION_LEVEL_DIR / f'tune_summary_{name}.json'
            part_summaries.append((name, out_summary))

            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                '--single_cand', name,
                '--out_episodes', str(out_episodes),
                '--out_summary', str(out_summary),
                '--device', device
            ]
            log_f = open(ACTION_LEVEL_DIR / f'tune_worker_{name}.log', 'w')
            sub_env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=sub_env)
            processes.append((proc, log_f, name))
            print(f"Launched Worker for [{name}] -> log: tune_worker_{name}.log", flush=True)

        for proc, log_f, name in processes:
            proc.wait()
            log_f.close()
            print(f"Worker [{name}] completed with exit code {proc.returncode}.", flush=True)
            if proc.returncode != 0:
                raise RuntimeError(f"Worker [{name}] failed! Check tune_worker_{name}.log")

        for name, summ_file in part_summaries:
            with open(summ_file, 'r') as f:
                results[name] = json.load(f)

    # Separate into Hysteresis and Repeat candidates
    hyst_cands = [v for v in results.values() if v['type'] == 'hysteresis']
    repeat_cands = [v for v in results.values() if v['type'] == 'repeat']

    # Selection rule: Highest SR -> Lowest CR -> Lower switch rate
    hyst_cands.sort(key=lambda x: (-x['success_rate'], x['collision_rate'], x['mean_grid_switch_rate']))
    best_hyst = hyst_cands[0]

    repeat_cands.sort(key=lambda x: (-x['success_rate'], x['collision_rate'], x['mean_grid_switch_rate']))
    best_repeat = repeat_cands[0]

    print("=" * 95)
    print("DEVELOPMENT BLOCK TUNING RESULTS:")
    print(f"{'Candidate':<20} | {'SR (%)':<7} | {'CR (%)':<7} | {'TR (%)':<7} | {'Time (s)':<8} | {'SwitchRate':<10} | {'Status'}")
    print("-" * 95)
    for c in list(results.values()):
        is_champ = (c['name'] in (best_hyst['name'], best_repeat['name']))
        tag = "WINNER (Hysteresis)" if c['name'] == best_hyst['name'] else ("WINNER (Repeat)" if c['name'] == best_repeat['name'] else "")
        print(f"{c['name']:<20} | {c['success_rate']:7.2f} | {c['collision_rate']:7.2f} | {c['timeout_rate']:7.2f} | {c['avg_time']:8.2f} | {c['mean_grid_switch_rate']:10.4f} | {tag}")
    print("=" * 95, flush=True)

    # Save locked parameters
    lock_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dev_block': {
            'scene': '20-human square_crossing',
            'case_range': [DEV_START_CASE, DEV_START_CASE + DEV_NUM_CASES - 1],
            'num_cases': DEV_NUM_CASES
        },
        'selection_rule': 'Highest Success Rate -> Lowest Collision Rate -> Lower Switch Rate',
        'locked_parameters': {
            'action_hysteresis': {
                'champion_name': best_hyst['name'],
                'alpha_a': best_hyst['alpha_a'],
                'metrics': best_hyst
            },
            'action_repeat': {
                'champion_name': best_repeat['name'],
                'k_repeat': best_repeat['k_repeat'],
                'metrics': best_repeat
            }
        },
        'all_evaluated_candidates': results
    }

    with open(PARAM_LOCK_FILE, 'w') as f:
        json.dump(lock_data, f, indent=2)

    with open(TUNE_RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"LOCKED PARAMETERS SAVED TO: {PARAM_LOCK_FILE}")
    print(f"  Action Hysteresis Locked alpha_a: {best_hyst['alpha_a']} (SR: {best_hyst['success_rate']}%, CR: {best_hyst['collision_rate']}%)")
    print(f"  Action Repeat Locked k:           {best_repeat['k_repeat']} (SR: {best_repeat['success_rate']}%, CR: {best_repeat['collision_rate']}%)")
    print(f"Total tuning wall time: {time.time() - t_start:.1f}s")
    print("=" * 95, flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--single_cand', type=str, default=None)
    parser.add_argument('--out_episodes', type=str, default=None)
    parser.add_argument('--out_summary', type=str, default=None)
    args = parser.parse_args()

    if args.single_cand:
        summary = evaluate_single_candidate(args.single_cand, args.out_episodes, device=args.device)
        if args.out_summary:
            with open(args.out_summary, 'w') as f:
                json.dump(summary, f, indent=2)
    else:
        run_tuning(device=args.device, workers=args.workers)
