#!/usr/bin/env python3
"""Event-Triggered Action Commitment & Recovery (ET-ACR) Benchmark on 50 Cases (90000-90049)

Evaluates Arm 3 (ET-ACR) on the confirmed KDARL benchmark:
- Base policy: Frozen GRU-VL (gru-t24-seed443.pth) with TTC-5 selection
- Safety filter: margin=0.2, risk_lambda=0.8, action_smoothing=0.3
- 20S square crossing: square_width=10.0, circle_radius=4.0, human_num=20, time_limit=50.0

Compares directly against:
- Arm 1: Base TTC-5 (SR=78.0%, CR=4.0%, TR=18.0%)
- Arm 2: Fixed 8-Step Commitment Heuristic (SR=6.0%, CR=90.0%, TR=4.0%)
"""

import sys
import os
import json
import time
from pathlib import Path
import configparser
import numpy as np
import torch

WORKSPACE = Path('/home/abc/workspace/autonomous_research_20261012')
os.chdir(str(WORKSPACE))
KDARL_CLEAN = WORKSPACE / 'kdarl_clean'
sys.path.insert(0, str(KDARL_CLEAN))

from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from run_selection_study_50cases import install_selection

def compute_trajectory_min_clearance(robot_radius, r_p0, r_v, humans, horizon_sec):
    min_dist = 999.0
    for h in humans:
        p_rel = np.array([r_p0[0] - h.px, r_p0[1] - h.py])
        v_rel = np.array([r_v[0] - h.vx, r_v[1] - h.vy])
        a = np.dot(v_rel, v_rel)
        b = 2.0 * np.dot(p_rel, v_rel)
        c = np.dot(p_rel, p_rel)
        t_min = np.clip(-b / (2.0 * a), 0.0, horizon_sec) if a > 1e-6 else 0.0
        d_min_sq = a * t_min**2 + b * t_min + c
        d_min = np.sqrt(max(d_min_sq, 0.0)) - robot_radius - h.radius
        if d_min < min_dist:
            min_dist = d_min
    return min_dist

def run_et_acr_benchmark(num_cases=50, commit_horizon=4, stagnation_window=16, start_step_threshold=40):
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
    install_selection(net, 'ttc5')

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
        'arm_name': 'Arm 3: Event-Triggered Action Commitment & Recovery (ET-ACR)',
        'success': 0,
        'collision': 0,
        'timeout': 0,
        'times': [],
        'cases': {},
        'triggers_per_case': {}
    }

    print(f"STARTING 50-CASE ET-ACR BENCHMARK (Horizon={commit_horizon}, Window={stagnation_window}, StartThreshold={start_step_threshold})...")
    start_time = time.time()

    for case_idx in range(num_cases):
        case_id = 90000 + case_idx
        obs, _ = env.reset(options={'test_case': case_id})
        net.reset_episode_stats()

        history_pos = []
        history_acts = []
        commit_steps_left = 0
        committed_action = None
        case_triggers = 0
        case_aborts = 0
        path_length = 0.0
        prev_pos = np.array([robot.px, robot.py], dtype=np.float64)
        init_dist = float(np.hypot(robot.px - robot.gx, robot.py - robot.gy))
        last_info = None

        for step in range(1, 202):
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            use_committed = False

            # Module 4: Safe Exit & Timely Abort (Margin = 0.22m)
            if commit_steps_left > 0 and committed_action is not None:
                dmin_curr = min(np.hypot(robot.px - h.px, robot.py - h.py) - robot.radius - h.radius for h in env.humans)
                if dmin_curr < 0.22:
                    case_aborts += 1
                    commit_steps_left = 0
                    committed_action = None
                else:
                    use_committed = True
                    action = committed_action
                    commit_steps_left -= 1

            if not use_committed:
                res = net.score_sarl_candidates(state)
                scores = res['scores'].numpy()
                cmds = res['commands']

                # Module 1: Online Stagnation Detector
                is_stagnated = False
                if step >= start_step_threshold and len(history_pos) >= stagnation_window:
                    recent_pos = history_pos[-stagnation_window:]
                    p_start = np.array(recent_pos[0])
                    p_end = np.array([robot.px, robot.py])
                    disp = float(np.linalg.norm(p_end - p_start))
                    p_len = sum(float(np.linalg.norm(np.array(recent_pos[i]) - np.array(recent_pos[i-1]))) for i in range(1, len(recent_pos)))

                    revs = 0
                    for i in range(len(history_acts) - stagnation_window + 1, len(history_acts)):
                        a1 = history_acts[i-1]
                        a2 = history_acts[i]
                        dot = a1[0]*a2[0] + a1[1]*a2[1]
                        m1 = np.hypot(a1[0], a1[1])
                        m2 = np.hypot(a2[0], a2[1])
                        if m1 > 0.1 and m2 > 0.1 and dot / (m1*m2) < -0.5:
                            revs += 1

                    if p_len > 1.6 and disp < 0.35 and (revs >= 2 or p_len / max(disp, 0.01) > 6.0):
                        is_stagnated = True

                # Module 2: Recovery Corridor Planner
                if is_stagnated:
                    rob_p = np.array([robot.px, robot.py])
                    goal_p = np.array([robot.gx, robot.gy])
                    to_goal = (goal_p - rob_p) / (np.linalg.norm(goal_p - rob_p) + 1e-6)
                    horizon_sec = commit_horizon * env.time_step

                    best_cmd = None
                    best_score = -999.0

                    for idx, cand_cmd in enumerate(cmds):
                        if scores[idx] < -100: continue
                        c_vec = np.array([cand_cmd.vx, cand_cmd.vy])
                        spd = float(np.linalg.norm(c_vec))
                        if spd < 0.3: continue

                        # Swept continuous clearance check: strict 0.22m clearance
                        min_clr = compute_trajectory_min_clearance(robot.radius, rob_p, c_vec, env.humans, horizon_sec)
                        if min_clr < 0.22: continue

                        p_end_cand = rob_p + horizon_sec * c_vec
                        prog = float(np.linalg.norm(rob_p - goal_p)) - float(np.linalg.norm(p_end_cand - goal_p))
                        align = float(np.dot(c_vec / (spd + 1e-6), to_goal))
                        if prog < 0.05: continue

                        score = scores[idx] + 2.0 * prog + 1.2 * min_clr + 0.6 * align
                        if score > best_score:
                            best_score = score
                            best_cmd = cand_cmd

                    # Module 3: Action Commitment
                    if best_cmd is not None:
                        case_triggers += 1
                        committed_action = best_cmd
                        commit_steps_left = commit_horizon
                        action = best_cmd
                        history_pos.clear()
                        history_acts.clear()
                    else:
                        action = cmds[scores.argmax().item()]
                else:
                    action = cmds[scores.argmax().item()]

            net._history.append(res['current_token'] if 'res' in locals() else net._history[-1])
            if net.test_action_smoothing > 0.0:
                net._last_action = action

            history_pos.append([robot.px, robot.py])
            history_acts.append([action.vx, action.vy])

            obs, reward, terminated, truncated, last_info = env.step(action)
            cur_pos = np.array([robot.px, robot.py], dtype=np.float64)
            path_length += float(np.linalg.norm(cur_pos - prev_pos))
            prev_pos = cur_pos

            if terminated or truncated:
                break

        event = last_info.get('event', 'nothing') if last_info else 'nothing'
        final_dist = float(np.hypot(robot.px - robot.gx, robot.py - robot.gy))
        net_prog = init_dist - final_dist

        if event in ('reach_goal', 'reachgoal', 'success') or env.robot.reached_destination():
            outcome = 'SUCCESS'
            results['times'].append(env.global_time)
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
            'triggers': case_triggers,
            'aborts': case_aborts
        }

        print(f"Case {case_id}: {outcome:<9} | t={env.global_time:5.2f}s | d_goal={final_dist:5.2f}m | Triggers={case_triggers} | Aborts={case_aborts}")

    elapsed = time.time() - start_time
    sr = 100.0 * results['success'] / num_cases
    cr = 100.0 * results['collision'] / num_cases
    tr = 100.0 * results['timeout'] / num_cases
    mean_time = np.mean(results['times']) if results['times'] else 0.0

    print(f"\n=================================================================")
    print(f"[ET-ACR BENCHMARK RESULTS (50 cases in {elapsed:.1f}s)]:")
    print(f"SR = {sr:.1f}% ({results['success']}/{num_cases})")
    print(f"CR = {cr:.1f}% ({results['collision']}/{num_cases})")
    print(f"TR = {tr:.1f}% ({results['timeout']}/{num_cases})")
    print(f"Mean Navigation Time (Success): {mean_time:.2f}s")
    print(f"=================================================================")

    # Save to json
    out_file = WORKSPACE / 'et_acr_50cases_benchmark_data.json'
    with open(out_file, 'w') as f:
        json.dump({
            'summary': {'sr': sr, 'cr': cr, 'tr': tr, 'mean_time': mean_time, 'num_cases': num_cases},
            'cases': results['cases']
        }, f, indent=2)
    print(f"Saved complete benchmark telemetry to {out_file}")
    return results

if __name__ == '__main__':
    run_et_acr_benchmark()
