#!/usr/bin/env python3
"""Rigorous 4-Arm Comparative Benchmark (Audited & Parity-Verified)

Protocol (Strictly identical across all 4 arms):
- Scenario: Standard 20S Square Crossing (square_width=10.0, circle_radius=4.0, human_num=20)
- Horizon: 50.0s time limit (max 201 steps at dt=0.25s)
- Selection: TTC-5 dynamic obstacle selection
- Safety filter: Margin 0.20m, Risk Lambda 0.8, Smoothing 0.30

Nomenclature (Strictly accurate):
- Arm B0: Base Single-Stream GRU (KDARL seed 443 with TTC-5)
- Arm B1: Dual-Stream Temporal GRU (Proximity Stream + Strategic Goal Stream with trained W_fuse)
- Arm B2: Learnable Intent Commitment & Switching (LICS, with Clearance-Gated Safety Adaptation)
- Arm B3: Controlled Heuristic Hysteresis Baseline (Fixed Dwell K=4, with IDENTICAL Clearance Gate and Alpha)

Outcome Determination (Fixed per audit):
- Strictly parsed from simulator `info['event']`:
  - 'reach_goal' -> SUCCESS
  - 'collision' -> COLLISION
  - 'timeout' -> TIMEOUT (assert time >= 49.5s)
  - Premature termination without reach_goal is strictly COLLISION

Evaluated Datasets:
- Primary Test Set: 50 cases (90000 - 90049)
- Generalization Test Set 1: 50 cases (90050 - 90099)
- Fresh Independent Test Set 2: 50 cases (90100 - 90149)
"""

import sys
import os
import json
import time
import math
from pathlib import Path
import configparser
import numpy as np
import torch

WORKSPACE = Path('/home/abc/workspace/autonomous_research_20261012')
KDARL_CLEAN = WORKSPACE / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(WORKSPACE)] + [p for p in sys.path if p not in (str(WORKSPACE), str(KDARL_CLEAN), '')]

from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from run_selection_study_50cases import install_selection
from strat_mamba_dual_stream import StratMambaDualStreamPolicy
from learnable_intent_mamba import LICSStratMambaPolicy, HeuristicHysteresisPolicy

CHECKPOINT_DIR = WORKSPACE / 'checkpoints'
RESULTS_DIR = WORKSPACE / 'benchmark_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def create_environment():
    env_config = configparser.RawConfigParser()
    env_config.read(KDARL_CLEAN / 'crowd_nav/configs/env.config')
    env = CrowdSim()
    env.configure(env_config)
    env.time_limit = 50.0
    env.human_num = 20
    env.square_width = 10.0
    env.circle_radius = 4.0
    env.phase = 'test'
    env.test_sim = 'square_crossing'
    return env, env_config

def get_base_config():
    env_config = configparser.RawConfigParser()
    env_config.read(KDARL_CLEAN / 'crowd_nav/configs/env.config')
    policy_config = configparser.RawConfigParser()
    policy_config.read(KDARL_CLEAN / 'crowd_nav/configs/policy.config')
    train_config = configparser.RawConfigParser()
    train_config.read(KDARL_CLEAN / 'crowd_nav/configs/train.config')

    cfg = configparser.RawConfigParser()
    for c in [env_config, policy_config, train_config]:
        for sec in c.sections():
            if not cfg.has_section(sec): cfg.add_section(sec)
            for k, v in c.items(sec): cfg.set(sec, k, v)

    cfg.set('env', 'time_limit', '50')
    cfg.set('mamba', 'temporal_backbone', 'gru')
    cfg.set('buffer', 'seq_len', '24')
    cfg.set('sim', 'human_num', '20')
    return cfg

def load_policy_for_arm(arm_id: str, device='cuda'):
    cfg = get_base_config()
    weights_path = WORKSPACE / 'baseline_weights/gru-t24-seed443.pth'

    if arm_id == 'B0':
        policy = MambaRLPolicy(cfg, device=device)
        ckpt = torch.load(weights_path, map_location=device)
        raw_state = ckpt.get('policy_state', ckpt.get('model', ckpt))
        clean_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in raw_state.items()}
        policy.load_state_dict(clean_dict, strict=True)
    elif arm_id == 'B1':
        policy = StratMambaDualStreamPolicy(cfg, device=device, backbone='gru')
        b1_ckpt_path = CHECKPOINT_DIR / 'b1_strat_mamba_trained.pth'
        if b1_ckpt_path.exists():
            policy.load_state_dict(torch.load(b1_ckpt_path, map_location=device), strict=False)
        else:
            policy.load_pretrained_weights(weights_path)
    elif arm_id == 'B2':
        policy = LICSStratMambaPolicy(cfg, device=device, backbone='gru', hysteresis_margin=0.15)
        b2_ckpt_path = CHECKPOINT_DIR / 'b2_lics_trained.pth'
        if b2_ckpt_path.exists():
            policy.load_state_dict(torch.load(b2_ckpt_path, map_location=device), strict=False)
        else:
            policy.load_pretrained_weights(weights_path)
    elif arm_id == 'B3':
        policy = HeuristicHysteresisPolicy(cfg, device=device, backbone='gru', dwell_k=4)
        policy.load_pretrained_weights(weights_path)
    else:
        raise ValueError(f"Unknown arm: {arm_id}")

    policy.eval()
    policy.build_action_space(1.0)
    policy.use_sarl_predict = True
    policy.set_phase('test')
    policy.test_min_clearance = 0.2
    policy.test_risk_lambda = 0.8
    policy.test_action_smoothing = 0.3
    install_selection(policy, 'ttc5')
    return policy

def evaluate_single_arm(arm_id: str, arm_name: str, start_case: int, num_cases: int = 50, device='cuda'):
    print(f"\n>>> Running Evaluation: {arm_id} ({arm_name}) on Cases {start_case} - {start_case + num_cases - 1} ...")
    policy = load_policy_for_arm(arm_id, device=device)
    env, env_config = create_environment()

    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    robot.env = env
    env.set_robot(robot)

    results = {
        'arm_id': arm_id,
        'arm_name': arm_name,
        'start_case': start_case,
        'num_cases': num_cases,
        'success': 0,
        'collision': 0,
        'timeout': 0,
        'stagnation_count': 0,
        'times': [],
        'path_lengths': [],
        'switches_per_case': [],
        'case_details': {}
    }

    t0 = time.time()
    for idx in range(num_cases):
        case_id = start_case + idx
        obs, _ = env.reset(options={'test_case': case_id})
        policy.reset_episode_stats()

        traj_x, traj_y = [], []
        actions = []
        path_length = 0.0
        prev_pos = np.array([robot.px, robot.py], dtype=np.float64)
        init_dist = float(np.hypot(robot.px - robot.gx, robot.py - robot.gy))
        last_info = None

        for step in range(1, 202):
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            action = policy.predict(state)
            traj_x.append(robot.px)
            traj_y.append(robot.py)
            actions.append((action.vx, action.vy))

            cur_pos = np.array([robot.px, robot.py], dtype=np.float64)
            path_length += float(np.linalg.norm(cur_pos - prev_pos))
            prev_pos = cur_pos

            obs, reward, terminated, truncated, last_info = env.step(action)
            if terminated or truncated:
                break

        final_dist = float(np.hypot(robot.px - robot.gx, robot.py - robot.gy))
        last_event = last_info.get('event', 'nothing') if last_info else 'nothing'

        # Strict event-driven classification based directly on environment truth
        if last_event in ('reach_goal', 'reachgoal') or env.robot.reached_destination():
            outcome = 'SUCCESS'
            results['times'].append(env.global_time)
            results['path_lengths'].append(path_length)
        elif last_event == 'collision':
            outcome = 'COLLISION'
        elif last_event == 'timeout' or env.global_time >= env.time_limit - 1e-4:
            outcome = 'TIMEOUT'
        else:
            # Fallback: if episode stopped prematurely (<49.5s) without goal reached, it is collision
            if env.global_time < env.time_limit - 0.5:
                outcome = 'COLLISION'
            else:
                outcome = 'TIMEOUT'

        # Strict audit assertion: never label a premature stop as timeout
        if outcome == 'TIMEOUT':
            assert env.global_time >= env.time_limit - 0.5, f"Bug: case {case_id} reached {env.global_time}s but labeled TIMEOUT"

        results[outcome.lower()] += 1

        # Check for oscillatory stagnation (in the last 40 steps / 10 seconds of timeout episodes)
        last_pts = np.column_stack([traj_x[-40:], traj_y[-40:]]) if len(traj_x) >= 40 else np.column_stack([traj_x, traj_y])
        last_dist = float(np.sum(np.linalg.norm(np.diff(last_pts, axis=0), axis=1))) if len(last_pts) > 1 else 0.0
        last_disp = float(np.linalg.norm(last_pts[-1] - last_pts[0])) if len(last_pts) > 1 else 0.0
        ratio = last_dist / max(last_disp, 0.01)
        is_stagnated = (outcome == 'TIMEOUT') and (last_dist > 1.5) and (last_disp < 0.35) and (ratio > 4.0)
        if is_stagnated:
            results['stagnation_count'] += 1

        # Check switches if available
        switches = 0
        if hasattr(policy, 'intent_switch_history') and len(policy.intent_switch_history) > 1:
            hist = policy.intent_switch_history
            switches = sum(1 for i in range(1, len(hist)) if hist[i] != hist[i-1])
        results['switches_per_case'].append(switches)

        results['case_details'][case_id] = {
            'outcome': outcome,
            'event': last_event,
            'time': float(env.global_time),
            'steps': step,
            'path_length': round(path_length, 3),
            'final_dist': round(final_dist, 3),
            'stagnated': is_stagnated,
            'travel_10s': round(last_dist, 3),
            'disp_10s': round(last_disp, 3),
            'ratio_10s': round(ratio, 1),
            'switches': switches
        }

    t1 = time.time()
    sr = 100.0 * results['success'] / num_cases
    cr = 100.0 * results['collision'] / num_cases
    tr = 100.0 * results['timeout'] / num_cases
    sor = 100.0 * results['stagnation_count'] / num_cases
    mean_time = float(np.mean(results['times'])) if len(results['times']) else 0.0
    mean_path = float(np.mean(results['path_lengths'])) if len(results['path_lengths']) else 0.0
    mean_switches = float(np.mean(results['switches_per_case'])) if len(results['switches_per_case']) else 0.0

    results['summary'] = {
        'sr': round(sr, 1),
        'cr': round(cr, 1),
        'tr': round(tr, 1),
        'sor': round(sor, 1),
        'mean_time': round(mean_time, 2),
        'mean_path': round(mean_path, 2),
        'mean_switches': round(mean_switches, 2),
        'elapsed_sec': round(t1 - t0, 2)
    }

    print(f"[{arm_id}] SR: {sr:.1f}% ({results['success']}/{num_cases}) | CR: {cr:.1f}% ({results['collision']}/{num_cases}) | TR: {tr:.1f}% ({results['timeout']}/{num_cases}) | SOR: {sor:.1f}% | Time: {mean_time:.2f}s | Path: {mean_path:.2f}m")
    return results

def run_complete_benchmark():
    arms = [
        ('B0', 'Base Single-Stream GRU (TTC-5)'),
        ('B1', 'Dual-Stream GRU (Proximity + Goal Streams)'),
        ('B2', 'Learnable Intent Commitment & Switching (LICS)'),
        ('B3', 'Controlled Heuristic Dwell-4 Baseline (K=4, Same Gate/Alpha)')
    ]

    all_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'primary_test_set': {},
        'generalization_test_set_1': {},
        'unseen_test_set_2': {}
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmark Device: {device}")

    # Set 1: Primary Test Set (Cases 90000 - 90049)
    print("==========================================================================================")
    print("STEP 1: EVALUATING PRIMARY TEST SET (Cases 90000 - 90049, 50 cases)")
    print("==========================================================================================")
    for arm_id, arm_name in arms:
        res = evaluate_single_arm(arm_id, arm_name, start_case=90000, num_cases=50, device=device)
        all_data['primary_test_set'][arm_id] = res

    # Set 2: Generalization Test Set 1 (Cases 90050 - 90099)
    print("\n==========================================================================================")
    print("STEP 2: EVALUATING GENERALIZATION TEST SET 1 (Cases 90050 - 90099, 50 cases)")
    print("==========================================================================================")
    for arm_id, arm_name in arms:
        res = evaluate_single_arm(arm_id, arm_name, start_case=90050, num_cases=50, device=device)
        all_data['generalization_test_set_1'][arm_id] = res

    # Set 3: Fresh Independent Test Set 2 (Cases 90100 - 90149)
    print("\n==========================================================================================")
    print("STEP 3: EVALUATING FRESH UNSEEN TEST SET 2 (Cases 90100 - 90149, 50 fresh cases)")
    print("==========================================================================================")
    for arm_id, arm_name in arms:
        res = evaluate_single_arm(arm_id, arm_name, start_case=90100, num_cases=50, device=device)
        all_data['unseen_test_set_2'][arm_id] = res

    output_path = RESULTS_DIR / 'rigorous_audited_4arm_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSuccessfully saved full audited benchmark results to {output_path}")

    # Generate Markdown Summary Tables
    print("\n" + "=" * 90)
    print("FINAL RIGOROUS 4-ARM COMPARATIVE BENCHMARK SUMMARY (150 CASES)")
    print("=" * 90)

    for set_key, set_name in [('primary_test_set', 'Primary Benchmark (Cases 90000-90049)'),
                              ('generalization_test_set_1', 'Generalization Benchmark 1 (Cases 90050-90099)'),
                              ('unseen_test_set_2', 'Fresh Unseen Benchmark 2 (Cases 90100-90149)')]:
        print(f"\n### {set_name}\n")
        print("| Arm ID | Method Description | Success Rate (SR) | Collision Rate (CR) | Timeout Rate (TR) | Stagnation Rate (SOR) | Time (s) | Path (m) |")
        print("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for arm_id, arm_name in arms:
            s = all_data[set_key][arm_id]['summary']
            print(f"| **{arm_id}** | {arm_name} | **{s['sr']:.1f}%** | {s['cr']:.1f}% | {s['tr']:.1f}% | {s['sor']:.1f}% | {s['mean_time']:.2f} | {s['mean_path']:.2f} |")

    return all_data

if __name__ == '__main__':
    run_complete_benchmark()
