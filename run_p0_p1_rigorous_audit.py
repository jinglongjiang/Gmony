#!/usr/bin/env python3
"""P0 & P1 Rigorous Audit:
Verify the 50 baseline cases (90000-90049) under exact protocols:
Arm 1: Official 50s limit (200 steps), with standard test-time safety filter (margin=0.2, risk=0.8, smooth=0.3)
Arm 2: 25s cutoff (100 steps), with standard test-time safety filter
Arm 3: Official 50s limit (200 steps), WITHOUT safety filter (margin=0.0, risk=0.0, smooth=0.0)

Extract full trajectory telemetry:
- final outcome (SUCCESS / COLLISION / TIMEOUT)
- global time & step count
- net progress (d_init - d_final)
- total path length traveled
- final distance to goal
- classification: Robot A (Frozen: net progress < 2.0m) vs Robot B (Progressing: net progress >= 2.0m)
"""

import sys
import json
from pathlib import Path
import configparser
import numpy as np
import torch

WORKSPACE = Path('/home/abc/workspace/autonomous_research_20261012')
sys.path.insert(0, str(WORKSPACE))

from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState

def run_audit(arm_name, time_limit_sec=50, use_safety_filter=True, num_cases=50):
    env_config = configparser.RawConfigParser()
    env_config.read(WORKSPACE / 'crowd_nav/configs/env.config')
    policy_config = configparser.RawConfigParser()
    policy_config.read(WORKSPACE / 'crowd_nav/configs/policy.config')
    train_config = configparser.RawConfigParser()
    train_config.read(WORKSPACE / 'crowd_nav/configs/train.config')

    cfg = configparser.RawConfigParser()
    for c in [env_config, policy_config, train_config]:
        for sec in c.sections():
            if not cfg.has_section(sec):
                cfg.add_section(sec)
            for k, v in c.items(sec):
                cfg.set(sec, k, v)

    cfg.set('env', 'time_limit', str(time_limit_sec))
    cfg.set('mamba', 'temporal_backbone', 'gru')
    cfg.set('buffer', 'seq_len', '24')
    cfg.set('sim', 'human_num', '20')

    net = MambaRLPolicy(cfg, device='cpu')
    weights_path = WORKSPACE / 'baseline_weights/gru-t24-seed443.pth'
    ckpt = torch.load(weights_path, map_location='cpu')
    clean_dict = {k.replace('module.', ''): v for k, v in (ckpt['model'] if 'model' in ckpt else ckpt).items()}
    net.load_state_dict(clean_dict, strict=False)
    net.eval()
    net.build_action_space(1.0)
    net.use_sarl_predict = True
    net.set_phase('test')

    if use_safety_filter:
        net.test_min_clearance = 0.2
        net.test_risk_lambda = 0.8
        net.test_action_smoothing = 0.3
    else:
        net.test_min_clearance = 0.0
        net.test_risk_lambda = 0.0
        net.test_action_smoothing = 0.0

    env = CrowdSim()
    env.configure(env_config)
    env.time_limit = time_limit_sec
    env.human_num = 20
    env.square_width = 10.0
    env.circle_radius = 4.0
    env.phase = 'test'
    env.test_sim = 'square_crossing'

    robot = Robot(env_config, 'robot')
    robot.set_policy(net)
    robot.env = env
    env.set_robot(robot)

    results = {
        'arm_name': arm_name,
        'time_limit_sec': time_limit_sec,
        'use_safety_filter': use_safety_filter,
        'success': 0,
        'collision': 0,
        'timeout': 0,
        'robot_a_frozen_timeouts': 0,
        'robot_b_late_timeouts': 0,
        'mean_nav_time_success': None,
        'cases': {}
    }

    max_steps = int(time_limit_sec / 0.25)
    success_times = []

    for case_idx in range(num_cases):
        case_id = 90000 + case_idx
        env.case_counter['test'] = case_id
        obs, _ = env.reset(options={'test_case': case_id})
        net.reset_episode_stats()
        done = False
        step = 0
        init_dist = float(np.hypot(robot.px - robot.gx, robot.py - robot.gy))
        path_length = 0.0
        prev_pos = np.array([robot.px, robot.py], dtype=np.float64)

        while not done and step < max_steps:
            step += 1
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            action = net.predict(state)
            obs, reward, terminated, truncated, info = env.step(action)
            cur_pos = np.array([robot.px, robot.py], dtype=np.float64)
            path_length += float(np.linalg.norm(cur_pos - prev_pos))
            prev_pos = cur_pos
            done = terminated or truncated

        final_dist = float(np.hypot(robot.px - robot.gx, robot.py - robot.gy))
        net_prog = init_dist - final_dist
        collision = any(np.linalg.norm([env.robot.px - h.px, env.robot.py - h.py]) < (env.robot.radius + h.radius) for h in env.humans)

        if env.robot.reached_destination():
            outcome = 'SUCCESS'
            success_times.append(float(env.global_time))
        elif collision:
            outcome = 'COLLISION'
        else:
            outcome = 'TIMEOUT'

        results[outcome.lower()] += 1

        is_robot_a = (outcome == 'TIMEOUT' and net_prog < 2.0)
        is_robot_b = (outcome == 'TIMEOUT' and net_prog >= 2.0)
        if is_robot_a:
            results['robot_a_frozen_timeouts'] += 1
        if is_robot_b:
            results['robot_b_late_timeouts'] += 1

        results['cases'][case_id] = {
            'outcome': outcome,
            'steps': step,
            'time': float(env.global_time),
            'init_dist': round(init_dist, 3),
            'final_dist': round(final_dist, 3),
            'net_progress': round(net_prog, 3),
            'path_length': round(path_length, 3),
            'type': 'FROZEN_A' if is_robot_a else ('LATE_B' if is_robot_b else outcome)
        }

    sr = 100.0 * results['success'] / num_cases
    cr = 100.0 * results['collision'] / num_cases
    tr = 100.0 * results['timeout'] / num_cases
    results['sr'] = sr
    results['cr'] = cr
    results['tr'] = tr
    if len(success_times):
        results['mean_nav_time_success'] = float(np.mean(success_times))

    print(f"\n[{arm_name}]:")
    print(f"  SR={sr:.1f}% ({results['success']}/{num_cases}), CR={cr:.1f}% ({results['collision']}/{num_cases}), TR={tr:.1f}% ({results['timeout']}/{num_cases})")
    if tr > 0:
        print(f"  Timeout Breakdown: Robot A (Frozen, progress<2m) = {results['robot_a_frozen_timeouts']} | Robot B (Progressing, progress>=2m) = {results['robot_b_late_timeouts']}")
    if len(success_times):
        print(f"  Mean Success Nav Time = {results['mean_nav_time_success']:.2f}s")

    return results

if __name__ == '__main__':
    print("=================================================================")
    print("STARTING RIGOROUS P0 & P1 AUDIT OVER 50 EPISODES (Cases 90000-90049)")
    print("=================================================================")

    # Arm 1: Official 50s limit (200 steps), Standard Filter
    res_50s_filter = run_audit("Arm 1: Official 50s (200 steps), Safety Filter ON", time_limit_sec=50, use_safety_filter=True, num_cases=50)

    # Arm 2: 25s cutoff (100 steps), Standard Filter
    res_25s_filter = run_audit("Arm 2: Cutoff 25s (100 steps), Safety Filter ON", time_limit_sec=25, use_safety_filter=True, num_cases=50)

    # Arm 3: Official 50s limit (200 steps), Safety Filter OFF
    res_50s_nofilter = run_audit("Arm 3: Official 50s (200 steps), Safety Filter OFF", time_limit_sec=50, use_safety_filter=False, num_cases=50)

    all_data = {
        'arm1_50s_filter': res_50s_filter,
        'arm2_25s_filter': res_25s_filter,
        'arm3_50s_nofilter': res_50s_nofilter
    }

    out_file = WORKSPACE / 'p0_p1_audit_data.json'
    with open(out_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"\nAll telemetry successfully saved to: {out_file}")
