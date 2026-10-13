#!/usr/bin/env python3
"""Cycle 2 Experiment: Closed-Loop Performance Comparison

Compare Baseline (static Euclidean top-5) vs. ACPS (Action-Conditioned Pedestrian Selection)
on 50 episodes of 20-human square crossing (20S, cases 90000-90049).

Hypothesis:
ACPS feeds the true constraint pedestrian into the candidate lookahead token,
eliminating blind spots during action evaluation and directly improving Success Rate (SR)
and reducing Collisions (CR).
"""

import sys
import copy
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
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized

def select_action_conditioned_top5(robot_state, human_states, candidate_vx, candidate_vy):
    """Action-Conditioned Selection (ACPS): conditions closing speed on candidate velocity."""
    if len(human_states) <= 5:
        return list(human_states)
    h = np.asarray([[x.px, x.py, x.vx, x.vy] for x in human_states], dtype=np.float64)
    dx = h[:, 0] - robot_state.px
    dy = h[:, 1] - robot_state.py
    dvx = h[:, 2] - candidate_vx
    dvy = h[:, 3] - candidate_vy
    d = np.sqrt(dx * dx + dy * dy + 1e-6)
    closing = -(dx * dvx + dy * dvy) / (d + 1e-6)
    rel_speed_sq = dvx * dvx + dvy * dvy + 1e-12
    t_star = np.clip(-(dx * dvx + dy * dvy) / rel_speed_sq, 0.0, 2.0)
    min_dist = np.sqrt((dx + t_star * dvx)**2 + (dy + t_star * dvy)**2)
    score = min_dist + 0.1 * np.where(closing > 0.1, d / (closing + 1e-6), d * 5)
    best_indices = np.argsort(score)[:5]
    return [human_states[i] for i in best_indices]

def patch_policy_with_acps(policy):
    """Patch policy._score_candidates_vectorized to use ACPS for each candidate action."""
    original_score = policy._score_candidates_vectorized

    def acps_score_candidates_vectorized(state, current_token=None):
        if policy.action_space is None:
            policy.build_action_space(state.self_state.v_pref)
        if current_token is None:
            current_34 = policy._build_joint_state_34(state.self_state, state.human_states)
            current_token = _batch_joint34_to_tokens_vectorized(current_34.reshape(1, -1))[0]
        base_hist = list(policy._history) + [current_token]
        commands = policy.executable_candidates()
        A = len(commands)
        rob = state.self_state

        next_humans = [policy.propagate(h, ActionXY(h.vx, h.vy)) for h in state.human_states]
        N = len(next_humans)
        cmd = np.array([[c.vx, c.vy] for c in commands], dtype=np.float64)
        nrp = np.array([rob.px, rob.py], dtype=np.float64)[None, :] + cmd * policy.time_step

        if N:
            hp = np.array([[h.px, h.py] for h in next_humans], dtype=np.float64)
            hr = np.array([h.radius for h in next_humans], dtype=np.float64)
            D = np.hypot(nrp[:, None, 0] - hp[None, :, 0],
                         nrp[:, None, 1] - hp[None, :, 1]) - rob.radius - hr[None, :]
            dmins = D.min(axis=1)
            collision = (D < 0.0).any(axis=1)
        else:
            dmins = np.full(A, np.inf)
            collision = np.zeros(A, dtype=bool)

        curr_dist = np.hypot(nrp[:, 0] - rob.gx, nrp[:, 1] - rob.gy)
        reaching = curr_dist < rob.radius
        prev_dist = float(np.linalg.norm((rob.px - rob.gx, rob.py - rob.gy)))
        speed = np.hypot(cmd[:, 0], cmd[:, 1])
        rew = policy.progress_reward * (prev_dist - curr_dist) + policy.time_penalty
        rew = rew + np.where(speed < 0.05, float(policy.stand_penalty), 0.0)
        rew = rew - np.where(dmins < policy.discomfort_dist,
                             policy.discomfort_penalty_factor * (policy.discomfort_dist - dmins) * policy.time_step, 0.0)
        rew = np.where(reaching, float(policy.success_reward), rew)
        rew = np.where(collision, float(policy.collision_penalty), rew)

        # 🔥 ACPS: Select humans tailored for EACH candidate action
        s34_list = []
        for a in commands:
            next_rob = policy.propagate(rob, a)
            tailored_humans = select_action_conditioned_top5(next_rob, next_humans, a.vx, a.vy)
            s34_list.append(policy._build_joint_state_34(next_rob, tailored_humans))

        s34 = np.stack(s34_list)
        next_tokens = _batch_joint34_to_tokens_vectorized(s34)

        L = policy.seq_len
        hist_arr = np.asarray(base_hist) if len(base_hist) else None
        if len(base_hist) + 1 >= L:
            pref = hist_arr[-(L - 1):] if L > 1 else hist_arr[:0]
            windows = np.concatenate([np.broadcast_to(pref[None], (A,) + pref.shape), next_tokens[:, None]], axis=1)
        else:
            pads = L - (len(base_hist) + 1)
            first = hist_arr[0] if hist_arr is not None and len(hist_arr) else next_tokens[0]
            pad_one = np.zeros_like(first) if policy.history_contract == 'legal-prefix' else first
            pad_block = np.broadcast_to(pad_one[None, None], (A, pads) + pad_one.shape)
            parts = [pad_block]
            if hist_arr is not None and len(hist_arr):
                parts.append(np.broadcast_to(hist_arr[None], (A,) + hist_arr.shape))
            parts.append(next_tokens[:, None])
            windows = np.concatenate(parts, axis=1)
        windows = np.ascontiguousarray(windows)
        return commands, current_token, torch.from_numpy(windows).float().to(policy.device), rew, dmins

    policy._score_candidates_vectorized = acps_score_candidates_vectorized

def evaluate_arm(arm_name, use_acps, num_cases=40):
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

    if use_acps:
        patch_policy_with_acps(net)

    env = CrowdSim()
    env.configure(env_config)
    env.human_num = 20
    env.square_width = 20.0
    env.circle_radius = 8.0
    env.phase = 'test'
    env.test_sim = 'square_crossing'

    robot = Robot(env_config, 'robot')
    robot.set_policy(net)
    robot.env = env
    env.set_robot(robot)

    results = {'success': 0, 'collision': 0, 'timeout': 0, 'time': [], 'min_clearance': []}

    print(f"\nEvaluating [{arm_name}] on {num_cases} episodes...")
    for case_idx in range(num_cases):
        case_id = 90000 + case_idx
        env.case_counter['test'] = case_id
        obs, _ = env.reset(options={'test_case': case_id})
        net.reset_episode_stats()
        done = False
        step = 0
        min_clr_ep = float('inf')

        while not done and step < 100:
            step += 1
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            action = net.predict(state)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if 'dmin' in info:
                min_clr_ep = min(min_clr_ep, info['dmin'])

        # Classify outcome
        collision = any(np.linalg.norm([env.robot.px - h.px, env.robot.py - h.py]) < (env.robot.radius + h.radius) for h in env.humans)
        if env.robot.reached_destination():
            outcome = 'success'
        elif collision:
            outcome = 'collision'
        else:
            outcome = 'timeout'

        results[outcome] += 1
        results['time'].append(env.global_time)
        results['min_clearance'].append(min_clr_ep)

    sr = 100.0 * results['success'] / num_cases
    cr = 100.0 * results['collision'] / num_cases
    tr = 100.0 * results['timeout'] / num_cases
    print(f"[{arm_name}] Results ({num_cases} cases): SR={sr:.1f}%, CR={cr:.1f}%, TR={tr:.1f}%, Avg Time={np.mean(results['time']):.2f}s")
    return results

def main():
    print("=" * 70)
    print("CYCLE 2: Closed-Loop Benchmark Comparison (20-human Square Crossing)")
    print("=" * 70)
    n = 30
    baseline_res = evaluate_arm("Baseline (Static TTC5)", use_acps=False, num_cases=n)
    acps_res = evaluate_arm("Ours (Action-Conditioned ACPS)", use_acps=True, num_cases=n)

    b_sr = 100.0 * baseline_res['success'] / n
    o_sr = 100.0 * acps_res['success'] / n
    b_cr = 100.0 * baseline_res['collision'] / n
    o_cr = 100.0 * acps_res['collision'] / n

    print("\n" + "=" * 70)
    print("CYCLE 2 VERDICT:")
    print(f"Baseline Success Rate: {b_sr:.1f}%, Collision Rate: {b_cr:.1f}%")
    print(f"Ours ACPS Success Rate: {o_sr:.1f}%, Collision Rate: {o_cr:.1f}%")
    print(f"Delta SR: {o_sr - b_sr:+.1f} pp, Delta CR: {o_cr - b_cr:+.1f} pp")
    
    passed = (o_sr >= b_sr) and (o_cr <= b_cr)
    print(f"Cycle 2 Status: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

if __name__ == '__main__':
    main()
