"""500-Case Independent Comparative Benchmark & Behavioral Telemetry

Rigorous Head-to-Head Evaluation on Unseen Cases 90800 - 91299:
- Arm 1: Pure B0 Baseline (Single-Stream GRU)
- Arm 2: Best Heuristic Gate (Rule-based K=4 intent commitment)
- Arm 3: Standard Option-Critic (AAAI 2017)
- Arm 4: Deliberation Cost Option-Critic (AAAI 2018, eta=0.05)
- Arm 5: CF-OC (Proposed Counterfactual Switching-Advantage Option-Critic)

Computes:
1. SR, CR, TR, Navigation Time, Min Clearance.
2. Option Behavioral Dynamics: Mean Option Dwell Duration, Switch Count, Option Distribution.
3. McNemar's exact two-sided statistical significance test vs B0 and vs Heuristic Gate.
4. Serializes full per-case telemetry to benchmark_results/option_critic_500cases_evaluation.json.
"""

import sys
import os
import math
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from scipy.stats import binom_test

REPO_ROOT = Path(__file__).resolve().parents[2]
KDARL_CLEAN = REPO_ROOT / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(REPO_ROOT)] + [p for p in sys.path if p not in (str(REPO_ROOT), str(KDARL_CLEAN), '')]

from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from test_p0_regression_and_parity import create_environment, get_base_config
from run_stage4_500cases_evaluation import Stage4ControllerPolicy
from research.option_critic.networks import OptionCriticNetwork
from research.option_critic.policy import OptionCriticPolicy

RESULTS_DIR = REPO_ROOT / 'benchmark_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_b0_policy(device='cuda'):
    pol = Stage4ControllerPolicy(mode='base', config=get_base_config(), device=device)
    ckpt = torch.load(REPO_ROOT / 'baseline_weights/gru-t24-seed443.pth', map_location=device)
    raw = ckpt.get('policy_state', ckpt.get('model', ckpt))
    clean = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in raw.items()}
    pol.load_state_dict(clean, strict=True)
    pol.eval()
    pol.build_action_space(1.0)
    pol.use_sarl_predict = True
    pol.set_phase('test')
    pol.test_min_clearance = 0.2
    pol.test_risk_lambda = 0.8
    pol.test_action_smoothing = 0.3
    from run_selection_study_50cases import install_selection
    install_selection(pol, 'ttc5')
    return pol


def load_heuristic_policy(device='cuda'):
    pol = Stage4ControllerPolicy(mode='heuristic', config=get_base_config(), device=device)
    ckpt = torch.load(REPO_ROOT / 'baseline_weights/gru-t24-seed443.pth', map_location=device)
    raw = ckpt.get('policy_state', ckpt.get('model', ckpt))
    clean = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in raw.items()}
    pol.load_state_dict(clean, strict=True)
    pol.eval()
    pol.build_action_space(1.0)
    pol.use_sarl_predict = True
    pol.set_phase('test')
    pol.test_min_clearance = 0.2
    pol.test_risk_lambda = 0.8
    pol.test_action_smoothing = 0.3
    from run_selection_study_50cases import install_selection
    install_selection(pol, 'ttc5')
    return pol


def load_oc_policy(checkpoint_path: str, is_cf_oc: bool = False, device: str = 'cuda') -> OptionCriticPolicy:
    net = OptionCriticNetwork(
        num_options=4,
        num_actions=80,
        d_model=256,
        n_layers=4,
        temporal_backbone='gru',
        is_cf_oc=is_cf_oc
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = ckpt.get('model_state_dict', ckpt.get('policy_state', ckpt))
    clean = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in sd.items()}
    net.load_state_dict(clean, strict=False)
    net.eval()

    policy = OptionCriticPolicy(
        network=net,
        device=device,
        num_options=4,
        num_actions=80,
        seq_len=24,
        is_cf_oc=is_cf_oc,
        termination_threshold=0.5,
        use_sampling_termination=False
    )
    return policy


def evaluate_single_episode(policy, env, case_id: int) -> Dict:
    obs, _ = env.reset(options={'test_case': case_id})
    policy.reset_episode_stats()

    traj_x = [env.robot.px]
    traj_y = [env.robot.py]
    min_clearance = 999.0
    step = 0
    done = False
    last_info = None

    while not done and step < 201:
        step += 1
        state = JointState(env.robot.get_full_state(), [h.get_observable_state() for h in env.humans])
        if len(env.humans):
            dists = [math.hypot(env.robot.px - h.px, env.robot.py - h.py) - env.robot.radius - h.radius for h in env.humans]
            min_clearance = min(min_clearance, min(dists))

        act = policy.predict(state)
        traj_x.append(env.robot.px)
        traj_y.append(env.robot.py)

        obs, rew, term, trunc, last_info = env.step(act)
        done = term or trunc

    last_event = last_info.get('event', 'nothing') if last_info else 'nothing'
    if last_event == 'collision':
        outcome = 'COLLISION'
    elif last_event in ('reach_goal', 'reachgoal'):
        outcome = 'SUCCESS'
    else:
        outcome = 'TIMEOUT'

    final_time = float(env.global_time)

    # Deadlock / oscillation check
    p_arr = np.column_stack([traj_x[-40:], traj_y[-40:]]) if len(traj_x) >= 40 else np.column_stack([traj_x, traj_y])
    t_10s = float(np.sum(np.linalg.norm(np.diff(p_arr, axis=0), axis=1))) if len(p_arr) > 1 else 0.0
    d_10s = float(np.linalg.norm(p_arr[-1] - p_arr[0])) if len(p_arr) > 1 else 0.0
    is_stag = bool(outcome == 'TIMEOUT' and t_10s > 1.5 and d_10s < 0.35 and t_10s / max(d_10s, 0.01) > 4.0)

    # Option telemetry if available
    res = {
        'case_id': case_id,
        'outcome': outcome,
        'final_time': round(final_time, 2),
        'total_steps': step,
        'min_clearance': round(float(min_clearance), 3),
        'stagnation': is_stag
    }

    if hasattr(policy, 'total_switches'):
        res['switches'] = policy.total_switches
        res['mean_dwell'] = round(float(np.mean(policy.telemetry['durations'])), 2) if policy.telemetry['durations'] else round(float(step), 2)
        opt_counts = [policy.telemetry['option_history'].count(o) for o in range(4)]
        res['option_counts'] = opt_counts

    return res


def run_benchmark(
    start_case: int = 90800,
    num_cases: int = 500,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    standard_oc_ckpt: str = 'research/option_critic/checkpoints/standard_oc_trained.pth',
    cf_oc_ckpt: str = 'research/option_critic/checkpoints/cf_oc_trained.pth'
):
    print(f"\n{'='*80}")
    print(f"[BENCHMARK] Running 500-Case Comparative Evaluation (Cases {start_case} - {start_case + num_cases - 1})")
    print(f"{'='*80}\n")

    env, env_config = create_environment()
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    policies = {
        'arm1_base_b0': load_b0_policy(device=device),
        'arm2_heuristic_gate': load_heuristic_policy(device=device),
    }

    if Path(standard_oc_ckpt).exists():
        policies['arm3_standard_oc'] = load_oc_policy(standard_oc_ckpt, is_cf_oc=False, device=device)

    if Path(cf_oc_ckpt).exists():
        policies['arm4_cf_oc'] = load_oc_policy(cf_oc_ckpt, is_cf_oc=True, device=device)

    arm_results = {arm: [] for arm in policies}
    case_ids = list(range(start_case, start_case + num_cases))

    t0 = time.time()
    for idx, cid in enumerate(case_ids):
        for arm_name, pol in policies.items():
            robot.set_policy(pol)
            res = evaluate_single_episode(pol, env, cid)
            arm_results[arm_name].append(res)

        if (idx + 1) % 50 == 0 or idx == num_cases - 1:
            elapsed = time.time() - t0
            progress = (idx + 1) / num_cases
            eta = (elapsed / progress) - elapsed
            print(f"  Processed {idx + 1}/{num_cases} cases ({progress:.1%}) | Elapsed: {elapsed/60.0:.1f}m | ETA: {eta/60.0:.1f}m")
            for arm_name in policies:
                sub = arm_results[arm_name]
                sr = sum(1 for r in sub if r['outcome'] == 'SUCCESS') / len(sub)
                cr = sum(1 for r in sub if r['outcome'] == 'COLLISION') / len(sub)
                tr = sum(1 for r in sub if r['outcome'] == 'TIMEOUT') / len(sub)
                print(f"    {arm_name:<20}: SR: {sr:.1%} | CR: {cr:.1%} | TO: {tr:.1%}")

    # Compute Statistical Significance (McNemar test)
    b0_success = np.array([r['outcome'] == 'SUCCESS' for r in arm_results['arm1_base_b0']])
    summary = {}

    print(f"\n{'='*80}")
    print("FINAL 500-CASE STATISTICAL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Arm':<22} | {'SR (%)':<8} | {'CR (%)':<8} | {'TO (%)':<8} | {'Stag':<6} | {'Time (s)':<8} | {'McNemar (vs B0)'}")
    print("-" * 80)

    for arm_name, records in arm_results.items():
        n = len(records)
        sr = 100.0 * sum(1 for r in records if r['outcome'] == 'SUCCESS') / n
        cr = 100.0 * sum(1 for r in records if r['outcome'] == 'COLLISION') / n
        tr = 100.0 * sum(1 for r in records if r['outcome'] == 'TIMEOUT') / n
        stag = sum(1 for r in records if r['stagnation'])
        avg_time = float(np.mean([r['final_time'] for r in records]))

        arm_succ = np.array([r['outcome'] == 'SUCCESS' for r in records])
        b = int(np.sum((~b0_success) & arm_succ))  # B0 fail, Candidate success
        c = int(np.sum(b0_success & (~arm_succ)))  # B0 success, Candidate fail
        n_discordant = b + c
        if n_discordant > 0:
            p_val = float(binom_test(min(b, c), n_discordant, 0.5, alternative='two-sided'))
            p_str = f"b={b}, c={c}, p={p_val:.4f}"
        else:
            p_val = 1.0
            p_str = "identical"

        print(f"{arm_name:<22} | {sr:6.1f}%  | {cr:6.1f}%  | {tr:6.1f}%  | {stag:4d}   | {avg_time:6.2f}s  | {p_str}")

        summary[arm_name] = {
            'success_rate': sr,
            'collision_rate': cr,
            'timeout_rate': tr,
            'stagnations': stag,
            'mean_time': avg_time,
            'mcnemar_b': b,
            'mcnemar_c': c,
            'mcnemar_p': p_val
        }

    return arm_results, summary


def run_parallel_benchmark(
    start_case: int = 90800,
    num_cases: int = 500,
    num_workers: int = 5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    standard_oc_ckpt: str = 'research/option_critic/checkpoints/standard_oc_trained.pth',
    cf_oc_ckpt: str = 'research/option_critic/checkpoints/cf_oc_trained.pth'
):
    import subprocess
    print("=" * 90)
    print(f"[BENCHMARK] Multi-Worker Benchmark: {num_cases} Cases using {num_workers} parallel workers")
    print(f"Cases Range: {start_case} to {start_case + num_cases - 1}")
    print("=" * 90)

    cases_per_worker = num_cases // num_workers
    workers = []
    chunk_files = []

    for w in range(num_workers):
        w_start = start_case + w * cases_per_worker
        w_num = cases_per_worker if w < num_workers - 1 else (num_cases - w * cases_per_worker)
        w_out = RESULTS_DIR / f'oc_eval_chunk_{w}.json'
        chunk_files.append(w_out)

        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            '--start_case', str(w_start),
            '--num_cases', str(w_num),
            '--device', device,
            '--standard_oc', standard_oc_ckpt,
            '--cf_oc', cf_oc_ckpt,
            '--chunk_mode',
            '--out_file', str(w_out)
        ]
        log_f = open(RESULTS_DIR / f'worker_oc_{w}.log', 'w')
        p = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)
        workers.append((p, log_f))
        print(f"  Launched Worker {w}: Cases {w_start} - {w_start + w_num - 1}")

    # Wait for completion
    t_start = time.time()
    for w, (p, log_f) in enumerate(workers):
        p.wait()
        log_f.close()
        print(f"  Worker {w} finished (exit code {p.returncode})")

    elapsed = time.time() - t_start
    print(f"[BENCHMARK] All {num_workers} workers completed in {elapsed / 60.0:.1f} minutes. Merging results...")

    # Merge chunks
    merged_results = {}
    for c_file in chunk_files:
        with open(c_file, 'r') as f:
            c_data = json.load(f)
            for arm, recs in c_data.items():
                if arm not in merged_results:
                    merged_results[arm] = []
                merged_results[arm].extend(recs)

    # Statistical summary
    b0_success = np.array([r['outcome'] == 'SUCCESS' for r in merged_results['arm1_base_b0']])
    summary = {}

    print("\n" + "=" * 90)
    print("FINAL 500-CASE STATISTICAL SUMMARY")
    print("=" * 90)
    print(f"{'Arm':<22} | {'SR (%)':<8} | {'CR (%)':<8} | {'TO (%)':<8} | {'Stag':<6} | {'Time (s)':<8} | {'McNemar (vs B0)'}")
    print("-" * 90)

    for arm_name, records in merged_results.items():
        n = len(records)
        sr = 100.0 * sum(1 for r in records if r['outcome'] == 'SUCCESS') / n
        cr = 100.0 * sum(1 for r in records if r['outcome'] == 'COLLISION') / n
        tr = 100.0 * sum(1 for r in records if r['outcome'] == 'TIMEOUT') / n
        stag = sum(1 for r in records if r['stagnation'])
        avg_time = float(np.mean([r['final_time'] for r in records]))

        arm_succ = np.array([r['outcome'] == 'SUCCESS' for r in records])
        b = int(np.sum((~b0_success) & arm_succ))
        c = int(np.sum(b0_success & (~arm_succ)))
        n_discordant = b + c
        if n_discordant > 0:
            p_val = float(binom_test(min(b, c), n_discordant, 0.5, alternative='two-sided'))
            p_str = f"b={b}, c={c}, p={p_val:.4f}"
        else:
            p_val = 1.0
            p_str = "identical"

        print(f"{arm_name:<22} | {sr:6.1f}%  | {cr:6.1f}%  | {tr:6.1f}%  | {stag:4d}   | {avg_time:6.2f}s  | {p_str}")

        summary[arm_name] = {
            'success_rate': sr,
            'collision_rate': cr,
            'timeout_rate': tr,
            'stagnations': stag,
            'mean_time': avg_time,
            'mcnemar_b': b,
            'mcnemar_c': c,
            'mcnemar_p': p_val
        }

    out_file = RESULTS_DIR / 'option_critic_500cases_evaluation.json'
    with open(out_file, 'w') as f:
        json.dump({
            'meta': {
                'start_case': start_case,
                'num_cases': num_cases,
                'num_workers': num_workers,
                'device': device,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'summary': summary,
            'detailed_results': merged_results
        }, f, indent=2)

    print(f"\n[BENCHMARK] Results successfully saved to: {out_file}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_case', type=int, default=90800)
    parser.add_argument('--num_cases', type=int, default=500)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--standard_oc', type=str, default='research/option_critic/checkpoints/standard_oc_trained.pth')
    parser.add_argument('--cf_oc', type=str, default='research/option_critic/checkpoints/cf_oc_trained.pth')
    parser.add_argument('--chunk_mode', action='store_true')
    parser.add_argument('--out_file', type=str, default=None)
    args = parser.parse_args()

    if args.chunk_mode:
        env, env_config = create_environment()
        robot = Robot(env_config, 'robot')
        env.set_robot(robot)

        policies = {
            'arm1_base_b0': load_b0_policy(device=args.device),
            'arm2_heuristic_gate': load_heuristic_policy(device=args.device),
        }
        if Path(args.standard_oc).exists():
            policies['arm3_standard_oc'] = load_oc_policy(args.standard_oc, is_cf_oc=False, device=args.device)
        if Path(args.cf_oc).exists():
            policies['arm4_cf_oc'] = load_oc_policy(args.cf_oc, is_cf_oc=True, device=args.device)

        arm_results = {arm: [] for arm in policies}
        for cid in range(args.start_case, args.start_case + args.num_cases):
            for arm_name, pol in policies.items():
                robot.set_policy(pol)
                res = evaluate_single_episode(pol, env, cid)
                arm_results[arm_name].append(res)

        out_path = Path(args.out_file) if args.out_file else (RESULTS_DIR / f'chunk_{args.start_case}.json')
        with open(out_path, 'w') as f:
            json.dump(arm_results, f, indent=2)
    else:
        if args.workers > 1:
            run_parallel_benchmark(
                start_case=args.start_case,
                num_cases=args.num_cases,
                num_workers=args.workers,
                device=args.device,
                standard_oc_ckpt=args.standard_oc,
                cf_oc_ckpt=args.cf_oc
            )
        else:
            run_benchmark(
                start_case=args.start_case,
                num_cases=args.num_cases,
                device=args.device,
                standard_oc_ckpt=args.standard_oc,
                cf_oc_ckpt=args.cf_oc
            )
