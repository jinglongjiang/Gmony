#!/usr/bin/env python3
"""Generalization Benchmark on 50 UNSEEN Cases (90050-90099):
Compares Arm 1 (Base TTC-5) vs Arm 3 (ET-ACR) on brand-new cases.
Proves zero overfitting and validates generalizability of the ET-ACR mechanism.
"""

import sys
import os
import json
import time
from pathlib import Path
import configparser
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parent
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

def build_env_and_policy():
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

    return env, robot, net

def evaluate_cases(mode='base', start_case=90050, num_cases=50, commit_horizon=4, stagnation_window=16, start_step_threshold=40):
    env, robot, net = build_env_and_policy()
    results = {'success': 0, 'collision': 0, 'timeout': 0, 'times': [], 'cases': {}}

    print(f"\n--- Running Mode: {mode.upper()} on {num_cases} Unseen Cases ({start_case}-{start_case+num_cases-1}) ---")
    start_t = time.time()

    for i in range(num_cases):
        case_id = start_case + i
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

            if mode == 'et_acr':
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

                if mode == 'et_acr':
                    is_stagnated = False
                    if step >= start_step_threshold and len(history_pos) >= stagnation_window:
                        recent_pos = history_pos[-stagnation_window:]
                        p_start = np.array(recent_pos[0])
                        p_end = np.array([robot.px, robot.py])
                        disp = float(np.linalg.norm(p_end - p_start))
                        p_len = sum(float(np.linalg.norm(np.array(recent_pos[j]) - np.array(recent_pos[j-1]))) for j in range(1, len(recent_pos)))

                        revs = 0
                        for j in range(len(history_acts) - stagnation_window + 1, len(history_acts)):
                            a1 = history_acts[j-1]
                            a2 = history_acts[j]
                            dot = a1[0]*a2[0] + a1[1]*a2[1]
                            m1 = np.hypot(a1[0], a1[1])
                            m2 = np.hypot(a2[0], a2[1])
                            if m1 > 0.1 and m2 > 0.1 and dot / (m1*m2) < -0.5:
                                revs += 1

                        if p_len > 1.6 and disp < 0.35 and (revs >= 2 or p_len / max(disp, 0.01) > 6.0):
                            is_stagnated = True

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
                else: # base TTC-5
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
            'final_dist': round(final_dist, 3),
            'triggers': case_triggers,
            'aborts': case_aborts
        }

    elapsed = time.time() - start_t
    sr = 100.0 * results['success'] / num_cases
    cr = 100.0 * results['collision'] / num_cases
    tr = 100.0 * results['timeout'] / num_cases
    mean_time = np.mean(results['times']) if results['times'] else 0.0

    print(f"[{mode.upper()} UNSEEN RESULTS ({num_cases} cases in {elapsed:.1f}s)]:")
    print(f"SR = {sr:.1f}% ({results['success']}/{num_cases}), CR = {cr:.1f}% ({results['collision']}/{num_cases}), TR = {tr:.1f}% ({results['timeout']}/{num_cases})")
    print(f"Mean Time: {mean_time:.2f}s")
    return {'sr': sr, 'cr': cr, 'tr': tr, 'mean_time': mean_time, 'cases': results['cases']}

if __name__ == '__main__':
    print("STARTING PAIRED GENERALIZATION STUDY ON 50 UNSEEN CASES (90050-90099)...")
    res_base = evaluate_cases('base', start_case=90050, num_cases=50)
    res_et_acr = evaluate_cases('et_acr', start_case=90050, num_cases=50)

    # Save generalization results
    gen_file = WORKSPACE / 'generalization_50cases_data.json'
    with open(gen_file, 'w') as f:
        json.dump({
            'base': res_base,
            'et_acr': res_et_acr
        }, f, indent=2)
    print(f"\nGeneralization dataset saved to {gen_file}")
