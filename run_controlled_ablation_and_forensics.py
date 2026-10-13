#!/usr/bin/env python3
"""Controlled Ablation & Trajectory Forensic Audit:
1. Arm 1-PureAblation:
   - Keep action_smoothing = 0.3 IDENTICAL to Arm 1.
   - Set test_min_clearance = 0.0, test_risk_lambda = 0.0.
   - Time limit = 50s (200 steps).
   - Compare strictly 1-to-1 against Arm 1 on cases 90000-90049.

2. Timeout Trajectory Forensics for Arm 1:
   - For all timeout cases in Arm 1, extract full step-by-step telemetry.
   - Deep dive into the final 10 seconds (t in [40.0, 50.0]s, steps 160-200):
     * dist_to_goal at t=40s vs t=50s
     * net displacement in last 10s
     * total path traveled in last 10s
     * average speed in last 10s
     * min distance to nearest human in last 10s
     * diagnosis: True Deadlock (speed < 0.05, no progress), Goal-mouth Oscillation, or Dynamic Crowding Detour.
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

def run_controlled_audit():
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

    cfg.set('env', 'time_limit', '50')
    cfg.set('mamba', 'temporal_backbone', 'gru')
    cfg.set('buffer', 'seq_len', '24')
    cfg.set('sim', 'human_num', '20')

    net = MambaRLPolicy(cfg, device='cpu')
    weights_path = WORKSPACE / 'baseline_weights/gru-t24-seed443.pth'
    ckpt = torch.load(weights_path, map_location='cpu')
    raw_state = ckpt.get('policy_state', ckpt.get('model', ckpt))
    clean_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in raw_state.items()}
    net.load_state_dict(clean_dict, strict=True)
    net.eval()
    net.build_action_space(1.0)
    net.use_sarl_predict = True
    net.set_phase('test')

    # SINGLE-VARIABLE CONTROLLED ABLATION:
    # Keep smoothing = 0.3, remove only safety margin and risk lambda!
    net.test_min_clearance = 0.0
    net.test_risk_lambda = 0.0
    net.test_action_smoothing = 0.3

    env = CrowdSim()
    env.configure(env_config)
    env.time_limit = 50
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
        'arm_name': 'Controlled Ablation: 50s, Safety Filter OFF (margin=0, risk=0), Smoothing ON (0.3)',
        'success': 0,
        'collision': 0,
        'timeout': 0,
        'mean_nav_time_success': None,
        'cases': {}
    }

    success_times = []
    num_cases = 50

    for case_idx in range(num_cases):
        case_id = 90000 + case_idx
        obs, _ = env.reset(options={'test_case': case_id})
        net.reset_episode_stats()
        done = False
        step = 0
        init_dist = float(np.hypot(robot.px - robot.gx, robot.py - robot.gy))
        path_length = 0.0
        prev_pos = np.array([robot.px, robot.py], dtype=np.float64)
        last_info = None

        while not done and step < 200:
            step += 1
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            action = net.predict(state)
            obs, reward, terminated, truncated, last_info = env.step(action)
            cur_pos = np.array([robot.px, robot.py], dtype=np.float64)
            path_length += float(np.linalg.norm(cur_pos - prev_pos))
            prev_pos = cur_pos
            done = terminated or truncated

        event = last_info.get('event', 'nothing') if last_info else 'nothing'
        final_dist = float(np.hypot(robot.px - robot.gx, robot.py - robot.gy))
        net_prog = init_dist - final_dist

        if event == 'reach_goal' or env.robot.reached_destination():
            outcome = 'SUCCESS'
            success_times.append(float(env.global_time))
        elif event == 'collision':
            outcome = 'COLLISION'
        else:
            outcome = 'TIMEOUT'

        results[outcome.lower()] += 1
        results['cases'][case_id] = {
            'outcome': outcome,
            'event': event,
            'steps': step,
            'time': float(env.global_time),
            'init_dist': round(init_dist, 3),
            'final_dist': round(final_dist, 3),
            'net_progress': round(net_prog, 3),
            'path_length': round(path_length, 3),
        }

    results['sr'] = 100.0 * results['success'] / num_cases
    results['cr'] = 100.0 * results['collision'] / num_cases
    results['tr'] = 100.0 * results['timeout'] / num_cases
    if len(success_times):
        results['mean_nav_time_success'] = float(np.mean(success_times))

    print("=================================================================")
    print("CONTROLLED ABLATION RESULTS (Only Safety Filter OFF, Smoothing=0.3 KEPT)")
    print("=================================================================")
    print(f"SR: {results['sr']:.1f}% ({results['success']}/{num_cases})")
    print(f"CR: {results['cr']:.1f}% ({results['collision']}/{num_cases})")
    print(f"TR: {results['tr']:.1f}% ({results['timeout']}/{num_cases})")
    if len(success_times):
        print(f"Mean Success Time: {results['mean_nav_time_success']:.2f}s")

    # Load Arm 1 from true_p0_p1_audit_data.json for direct paired comparison
    with open(WORKSPACE / 'true_p0_p1_audit_data.json') as f:
        prior_data = json.load(f)
    arm1 = prior_data['arm1_50s_filter']['cases']

    print("\n--- PAIRED COMPARISON WITH ARM 1 (Filter ON vs Filter OFF, Smoothing=0.3 constant) ---")
    diffs = []
    for cid, cinfo in results['cases'].items():
        cid_str = str(cid)
        arm1_info = arm1[cid_str]
        if cinfo['outcome'] != arm1_info['outcome']:
            diffs.append((cid_str, arm1_info['outcome'], cinfo['outcome'], arm1_info['time'], cinfo['time']))

    print(f"Number of cases differing: {len(diffs)} / {num_cases}")
    for d in diffs:
        print(f"  Case {d[0]}: Arm 1 (Filter ON) = {d[1]} (t={d[3]:.2f}s) | Controlled (Filter OFF) = {d[2]} (t={d[4]:.2f}s)")

    # Save to json
    with open(WORKSPACE / 'controlled_ablation_data.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results

def run_timeout_forensics():
    """Extract full trajectory details for all 7 timeout cases of Arm 1."""
    with open(WORKSPACE / 'true_p0_p1_audit_data.json') as f:
        prior_data = json.load(f)
    arm1 = prior_data['arm1_50s_filter']['cases']
    timeout_ids = [int(k) for k, v in arm1.items() if v['outcome'] == 'TIMEOUT']

    print("\n=================================================================")
    print(f"ARM 1 TIMEOUT TRAJECTORY FORENSICS (Cases: {timeout_ids})")
    print("=================================================================")

    env_config = configparser.RawConfigParser()
    env_config.read(WORKSPACE / 'crowd_nav/configs/env.config')
    policy_config = configparser.RawConfigParser()
    policy_config.read(WORKSPACE / 'crowd_nav/configs/policy.config')
    train_config = configparser.RawConfigParser()
    train_config.read(WORKSPACE / 'crowd_nav/configs/train.config')

    cfg = configparser.RawConfigParser()
    for c in [env_config, policy_config, train_config]:
        for sec in c.sections():
            if not cfg.has_section(sec): cfg.add_section(sec)
            for k, v in c.items(sec): cfg.set(sec, k, v)

    cfg.set('env', 'time_limit', '50')
    cfg.set('mamba', 'temporal_backbone', 'gru')
    cfg.set('buffer', 'seq_len', '24')
    cfg.set('sim', 'human_num', '20')

    net = MambaRLPolicy(cfg, device='cpu')
    weights_path = WORKSPACE / 'baseline_weights/gru-t24-seed443.pth'
    ckpt = torch.load(weights_path, map_location='cpu')
    raw_state = ckpt.get('policy_state', ckpt.get('model', ckpt))
    clean_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in raw_state.items()}
    net.load_state_dict(clean_dict, strict=True)
    net.eval()
    net.build_action_space(1.0)
    net.use_sarl_predict = True
    net.set_phase('test')
    net.test_min_clearance = 0.2
    net.test_risk_lambda = 0.8
    net.test_action_smoothing = 0.3

    env = CrowdSim()
    env.configure(env_config)
    env.time_limit = 50
    env.human_num = 20
    env.square_width = 10.0
    env.circle_radius = 4.0
    env.phase = 'test'
    env.test_sim = 'square_crossing'

    robot = Robot(env_config, 'robot')
    robot.set_policy(net)
    robot.env = env
    env.set_robot(robot)

    forensics = {}

    for case_id in timeout_ids:
        obs, _ = env.reset(options={'test_case': case_id})
        net.reset_episode_stats()
        trajectory = []
        done = False
        step = 0

        while not done and step < 200:
            step += 1
            t_curr = float(env.global_time)
            pos_before = np.array([robot.px, robot.py], dtype=np.float64)
            d_goal_before = float(np.hypot(robot.px - robot.gx, robot.py - robot.gy))
            
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            action = net.predict(state)
            obs, reward, terminated, truncated, info = env.step(action)
            pos_after = np.array([robot.px, robot.py], dtype=np.float64)
            d_goal_after = float(np.hypot(robot.px - robot.gx, robot.py - robot.gy))
            speed = float(np.hypot(action.vx, action.vy))

            trajectory.append({
                'step': step,
                'time': round(t_curr, 2),
                'px': round(pos_after[0], 3),
                'py': round(pos_after[1], 3),
                'vx': round(action.vx, 3),
                'vy': round(action.vy, 3),
                'speed': round(speed, 3),
                'd_goal': round(d_goal_after, 3),
                'dmin': round(info.get('dmin', 999.0), 3)
            })
            done = terminated or truncated

        # Analyze the final 10 seconds: t in [40.0, 50.0]s -> steps 161 to 200 (last 40 steps)
        last_40 = trajectory[160:] if len(trajectory) >= 160 else trajectory
        p_start_last10 = np.array([last_40[0]['px'], last_40[0]['py']])
        p_end_last10 = np.array([last_40[-1]['px'], last_40[-1]['py']])
        net_disp_last10 = float(np.linalg.norm(p_end_last10 - p_start_last10))
        
        path_last10 = sum(
            float(np.hypot(last_40[i]['px'] - last_40[i-1]['px'], last_40[i]['py'] - last_40[i-1]['py']))
            for i in range(1, len(last_40))
        )
        speeds_last10 = [pt['speed'] for pt in last_40]
        mean_speed_last10 = float(np.mean(speeds_last10))
        dmins_last10 = [pt['dmin'] for pt in last_40]
        min_dmin_last10 = float(np.min(dmins_last10))
        d_goal_40s = last_40[0]['d_goal']
        d_goal_50s = last_40[-1]['d_goal']

        # Oscillation ratio: path traveled / net displacement
        oscillation_ratio = (path_last10 / max(net_disp_last10, 0.01))

        # Diagnosis
        if mean_speed_last10 < 0.08:
            diag = "TRUE_FREEZE (Robot stationary/standstill, v < 0.08m/s)"
        elif oscillation_ratio > 4.0:
            diag = f"DEADLOCK_OSCILLATION (Moving but trapped in place: path={path_last10:.2f}m, disp={net_disp_last10:.2f}m, ratio={oscillation_ratio:.1f})"
        elif min_dmin_last10 < 0.25:
            diag = f"HUMAN_BLOCKADE (Pushed away or pinned by close humans: min_dmin={min_dmin_last10:.2f}m)"
        else:
            diag = f"SLOW_PROGRESS (Progressing steadily but ran out of time: d_goal={d_goal_50s:.2f}m)"

        print(f"\n[Case {case_id}]:")
        print(f"  Init -> Final Pos: (0.0, -4.0) -> ({last_40[-1]['px']:.2f}, {last_40[-1]['py']:.2f}) | Goal: (0.0, 4.0)")
        print(f"  Full Episode Net Progress: {8.0 - d_goal_50s:.2f}m / 8.0m | Final d_goal: {d_goal_50s:.2f}m")
        print(f"  Last 10 Seconds (t=40~50s):")
        print(f"    d_goal: {d_goal_40s:.2f}m -> {d_goal_50s:.2f}m")
        print(f"    Net Displacement: {net_disp_last10:.2f}m | Total Path: {path_last10:.2f}m")
        print(f"    Mean Speed: {mean_speed_last10:.2f}m/s | Min Clearance: {min_dmin_last10:.2f}m")
        print(f"  => Diagnosis: {diag}")

        forensics[case_id] = {
            'init_dist': 8.0,
            'final_d_goal': d_goal_50s,
            'net_progress': round(8.0 - d_goal_50s, 3),
            'last_10s_d_goal_start': d_goal_40s,
            'last_10s_d_goal_end': d_goal_50s,
            'last_10s_net_displacement': round(net_disp_last10, 3),
            'last_10s_path_length': round(path_last10, 3),
            'last_10s_mean_speed': round(mean_speed_last10, 3),
            'last_10s_min_clearance': round(min_dmin_last10, 3),
            'diagnosis': diag
        }

    with open(WORKSPACE / 'timeout_forensics_data.json', 'w') as f:
        json.dump(forensics, f, indent=2)
    print("\nTimeout forensics data successfully saved to: timeout_forensics_data.json")

if __name__ == '__main__':
    run_controlled_audit()
    run_timeout_forensics()
