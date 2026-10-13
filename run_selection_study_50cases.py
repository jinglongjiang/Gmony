#!/usr/bin/env python3
"""Definitive Human Selection Comparison on 50 Cases (90000-90049):
Uses the clean, unpolluted KDARL codebase (kdarl_clean/) and gru-t24-seed443.pth.

Evaluates 3 arms under exact protocol (50s limit, max 201 steps, safety filter ON, smoothing 0.3):
1. Arm First-5: KDARL original unpatched native default (first 5 humans in state).
2. Arm Nearest-5: 5 Euclidean-distance nearest humans to robot.
3. Arm TTC-5: Official KDARL TTC5 selection (lowest TTC closing humans).

Specifically tracks:
- Global: SR, CR, TR, Mean Navigation Time
- The 4 goal-mouth oscillatory stagnation cases: 90000, 90010, 90024, 90034
- Full final 10s trajectory forensics on all remaining timeout cases under TTC5.
"""

import sys
import json
import time
from pathlib import Path
import configparser
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parent
KDARL_CLEAN = WORKSPACE / 'kdarl_clean'
sys.path.insert(0, str(KDARL_CLEAN))

from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState

def select_indices(robot, humans, rule):
    if rule == 'first5' or len(humans) <= 5:
        return list(range(min(5, len(humans))))
    h = np.array([[x.px, x.py, x.vx, x.vy, x.radius] for x in humans], dtype=np.float64)
    if rule == 'nearest5':
        score = np.hypot(h[:, 0] - robot.px, h[:, 1] - robot.py)
    elif rule == 'ttc5':
        rp = np.array([robot.px, robot.py, robot.vx, robot.vy], dtype=np.float32).astype(np.float64)
        rx, ry = h[:, 0] - rp[0], h[:, 1] - rp[1]
        rvx, rvy = h[:, 2] - rp[2], h[:, 3] - rp[3]
        distances = np.sqrt(rx ** 2 + ry ** 2 + 1e-6)
        closing = -(rx * rvx + ry * rvy) / (distances + 1e-6)
        with np.errstate(divide='ignore', invalid='ignore'):
            score = np.where(closing > .1, distances / (closing + 1e-6), distances * 10)
    else:
        raise ValueError(rule)
    return sorted(np.argsort(score)[:5].tolist())

def install_selection(net, rule):
    original_build = net._build_joint_state_34
    original_fast = net.score_sarl_candidates
    net.selection_rule = rule
    net.selection_indices = None

    def build(robot, humans):
        assert net.selection_indices is not None
        return original_build(robot, [humans[i] for i in net.selection_indices])

    def score(state):
        assert net._phase in ('test', 'val', 'eval'), 'Selection is evaluation-only'
        net.selection_indices = select_indices(state.self_state, state.human_states, net.selection_rule)
        result = original_fast(state)
        return result

    net._build_joint_state_34 = build
    net.score_sarl_candidates = score

def evaluate_selection_arm(rule, num_cases=50):
    env_config = configparser.RawConfigParser()
    env_config.read(KDARL_CLEAN / 'crowd_nav/configs/env.config')
    policy_config = configparser.RawConfigParser()
    policy_config.read(KDARL_CLEAN / 'crowd_nav/configs/policy.config')
    train_config = configparser.RawConfigParser()
    train_config.read(KDARL_CLEAN / 'crowd_nav/configs/train.config')

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
    net.test_min_clearance = 0.2
    net.test_risk_lambda = 0.8
    net.test_action_smoothing = 0.3

    install_selection(net, rule)

    env = CrowdSim()
    env.configure(env_config)
    env.time_limit = 50.0
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
        'rule': rule,
        'success': 0,
        'collision': 0,
        'timeout': 0,
        'mean_nav_time_success': None,
        'cases': {},
        'trajectories_timeout': {}
    }

    success_times = []

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
        ep_traj = []

        # Run up to 201 steps so step 201 triggers env global_time >= 50.0 check
        for step in range(1, 202):
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            action = net.predict(state)
            obs, reward, terminated, truncated, last_info = env.step(action)
            cur_pos = np.array([robot.px, robot.py], dtype=np.float64)
            path_length += float(np.linalg.norm(cur_pos - prev_pos))
            prev_pos = cur_pos

            ep_traj.append({
                'step': step,
                'time': round(float(env.global_time), 2),
                'px': round(float(robot.px), 3),
                'py': round(float(robot.py), 3),
                'vx': round(float(action.vx), 3),
                'vy': round(float(action.vy), 3),
                'speed': round(float(np.hypot(action.vx, action.vy)), 3),
                'd_goal': round(float(np.hypot(robot.px - robot.gx, robot.py - robot.gy)), 3),
                'dmin': round(float(last_info.get('dmin', 999.0)), 3)
            })

            done = terminated or truncated
            if done:
                break

        event = last_info.get('event', 'nothing') if last_info else 'nothing'
        final_dist = float(np.hypot(robot.px - robot.gx, robot.py - robot.gy))
        net_prog = init_dist - final_dist

        if event in ('reach_goal', 'reachgoal', 'success') or env.robot.reached_destination():
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

        if outcome == 'TIMEOUT':
            results['trajectories_timeout'][case_id] = ep_traj

    results['sr'] = 100.0 * results['success'] / num_cases
    results['cr'] = 100.0 * results['collision'] / num_cases
    results['tr'] = 100.0 * results['timeout'] / num_cases
    if len(success_times):
        results['mean_nav_time_success'] = float(np.mean(success_times))

    print(f"\n=================================================================")
    print(f"[{rule.upper()} RESULTS]: SR={results['sr']:.1f}% ({results['success']}/{num_cases}), CR={results['cr']:.1f}% ({results['collision']}/{num_cases}), TR={results['tr']:.1f}% ({results['timeout']}/{num_cases})")
    if len(success_times):
        print(f"Mean Navigation Time (Success): {results['mean_nav_time_success']:.2f}s")
    print(f"=================================================================")

    return results

def analyze_stagnation_forensics(rule_name, timeout_trajectories):
    print(f"\n--- Trajectory Stagnation Forensics for {rule_name.upper()} Timeouts ---")
    forensics = {}
    for case_id, traj in timeout_trajectories.items():
        last_40 = traj[-40:] if len(traj) >= 40 else traj
        p_start = np.array([last_40[0]['px'], last_40[0]['py']])
        p_end = np.array([last_40[-1]['px'], last_40[-1]['py']])
        net_disp = float(np.linalg.norm(p_end - p_start))
        path_len = sum(
            float(np.hypot(last_40[i]['px'] - last_40[i-1]['px'], last_40[i]['py'] - last_40[i-1]['py']))
            for i in range(1, len(last_40))
        )
        speeds = [pt['speed'] for pt in last_40]
        mean_speed = float(np.mean(speeds))
        d_goal_start = last_40[0]['d_goal']
        d_goal_end = last_40[-1]['d_goal']
        osc_ratio = path_len / max(net_disp, 0.01)

        is_goal_mouth = (d_goal_end < 1.5)
        is_oscillatory = (osc_ratio > 10.0 and mean_speed > 0.15)

        category = "GOAL_MOUTH_OSCILLATION" if (is_goal_mouth and is_oscillatory) else (
            "CENTER_OSCILLATION" if is_oscillatory else (
                "TRUE_STANDSTILL" if mean_speed < 0.08 else "DRIFT_OR_WANDER"
            )
        )

        print(f"Case {case_id}: d_goal={d_goal_end:.2f}m, Last10s Path={path_len:.2f}m, Disp={net_disp:.2f}m, Speed={mean_speed:.2f}m/s, Ratio={osc_ratio:.1f} -> {category}")
        forensics[case_id] = {
            'final_d_goal': d_goal_end,
            'last10s_path': round(path_len, 3),
            'last10s_disp': round(net_disp, 3),
            'last10s_mean_speed': round(mean_speed, 3),
            'oscillation_ratio': round(osc_ratio, 1),
            'category': category
        }
    return forensics

if __name__ == '__main__':
    print("STARTING DEFINITIVE HUMAN SELECTION COMPARISON (First-5 vs Nearest-5 vs TTC-5)...")

    # 1. TTC-5
    res_ttc5 = evaluate_selection_arm('ttc5', num_cases=50)
    forensics_ttc5 = analyze_stagnation_forensics('ttc5', res_ttc5['trajectories_timeout'])

    # 2. First-5
    res_first5 = evaluate_selection_arm('first5', num_cases=50)
    forensics_first5 = analyze_stagnation_forensics('first5', res_first5['trajectories_timeout'])

    # 3. Nearest-5
    res_nearest5 = evaluate_selection_arm('nearest5', num_cases=50)
    forensics_nearest5 = analyze_stagnation_forensics('nearest5', res_nearest5['trajectories_timeout'])

    # Focus analysis on the 4 key cases: 90000, 90010, 90024, 90034
    focus_cases = [90000, 90010, 90024, 90034]
    print("\n=================================================================")
    print("DEEP DIVE: THE 4 GOAL-MOUTH LOW-EFFICIENCY CASES (90000, 90010, 90024, 90034)")
    print("=================================================================")
    for cid in focus_cases:
        f_res = res_first5['cases'][cid]
        n_res = res_nearest5['cases'][cid]
        t_res = res_ttc5['cases'][cid]
        print(f"\nCase {cid}:")
        print(f"  First-5   : Outcome={f_res['outcome']}, Final d_goal={f_res['final_dist']}m, Time={f_res['time']:.2f}s, Path={f_res['path_length']:.2f}m")
        print(f"  Nearest-5 : Outcome={n_res['outcome']}, Final d_goal={n_res['final_dist']}m, Time={n_res['time']:.2f}s, Path={n_res['path_length']:.2f}m")
        print(f"  TTC-5     : Outcome={t_res['outcome']}, Final d_goal={t_res['final_dist']}m, Time={t_res['time']:.2f}s, Path={t_res['path_length']:.2f}m")

    # Save complete dataset
    all_out = {
        'ttc5': {
            'summary': {'sr': res_ttc5['sr'], 'cr': res_ttc5['cr'], 'tr': res_ttc5['tr'], 'time': res_ttc5['mean_nav_time_success']},
            'cases': res_ttc5['cases'],
            'forensics': forensics_ttc5
        },
        'first5': {
            'summary': {'sr': res_first5['sr'], 'cr': res_first5['cr'], 'tr': res_first5['tr'], 'time': res_first5['mean_nav_time_success']},
            'cases': res_first5['cases'],
            'forensics': forensics_first5
        },
        'nearest5': {
            'summary': {'sr': res_nearest5['sr'], 'cr': res_nearest5['cr'], 'tr': res_nearest5['tr'], 'time': res_nearest5['mean_nav_time_success']},
            'cases': res_nearest5['cases'],
            'forensics': forensics_nearest5
        }
    }

    out_path = WORKSPACE / 'selection_study_50cases.json'
    with open(out_path, 'w') as f:
        json.dump(all_out, f, indent=2)
    print(f"\nComplete selection study data successfully saved to: {out_path}")
