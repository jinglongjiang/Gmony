#!/usr/bin/env python3
"""Test Combined Shield (ACPS + Progress Incentive + Swept VO Filter) on Case 90000."""

import sys
from pathlib import Path
import configparser
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized

def compute_swept_clearance(rob_pos, rob_radius, cmd_vel, humans, horizon=1.0):
    """Compute exact continuous-time minimum clearance for candidate action over [0, horizon]."""
    vx, vy = cmd_vel
    min_clr = float('inf')

    for h in humans:
        rx = h.px - rob_pos[0]
        ry = h.py - rob_pos[1]
        rvx = h.vx - vx
        rvy = h.vy - vy
        comb_r = rob_radius + h.radius

        r_dot_rv = rx * rvx + ry * rvy
        rv_sq = rvx * rvx + rvy * rvy

        d0 = np.hypot(rx, ry) - comb_r
        if d0 < min_clr:
            min_clr = d0

        if rv_sq > 1e-8:
            t_cpa = -r_dot_rv / rv_sq
            if 0 < t_cpa <= horizon:
                d_cpa = np.hypot(rx + t_cpa * rvx, ry + t_cpa * rvy) - comb_r
                if d_cpa < min_clr:
                    min_clr = d_cpa
            elif t_cpa > horizon:
                d_end = np.hypot(rx + horizon * rvx, ry + horizon * rvy) - comb_r
                if d_end < min_clr:
                    min_clr = d_end

    return min_clr

def select_action_conditioned_top5(robot_state, human_states, candidate_vx, candidate_vy):
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
        rew = rew + np.where(reaching, float(policy.success_reward), 0.0)
        rew = np.where(collision, float(policy.collision_penalty), rew)

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

def run_test(case_id=90000, alpha=0.15, horizon=1.0, margin=0.10):
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
    net.progress_reward = alpha

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

    env.case_counter['test'] = case_id
    obs, _ = env.reset(options={'test_case': case_id})
    net.reset_episode_stats()
    done = False
    step = 0
    init_dist = np.hypot(robot.px - robot.gx, robot.py - robot.gy)

    while not done and step < 100:
        step += 1
        state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
        
        # Candidate scoring with Swept-VO Filter
        commands, current_token, next_states_tensor, rew_batch, dmins_batch = \
            net._score_candidates_vectorized(state)

        with torch.no_grad():
            next_values = net.forward_candidate_values(next_states_tensor)

        rewards_tensor = torch.tensor(rew_batch.tolist(), device=net.device)
        total_values = net.candidate_action_values(state, commands, rewards_tensor, next_values)

        # Swept-VO filter
        rob_pos = (robot.px, robot.py)
        swept_clrs = np.array([compute_swept_clearance(rob_pos, robot.radius, (c.vx, c.vy), env.humans, horizon=horizon) for c in commands])
        safe_mask = swept_clrs >= margin

        if safe_mask.any():
            # Filter unsafe actions
            total_values = total_values.masked_fill(~torch.as_tensor(safe_mask, device=net.device), -1e9)
            best_idx = total_values.argmax().item()
            action = commands[best_idx]
        else:
            # If all actions within horizon violate margin, pick action maximizing swept clearance
            best_idx = int(np.argmax(swept_clrs))
            action = commands[best_idx]

        net._history.append(current_token)
        net._last_action = action

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    final_dist = np.hypot(robot.px - robot.gx, robot.py - robot.gy)
    collision = any(np.linalg.norm([env.robot.px - h.px, env.robot.py - h.py]) < (env.robot.radius + h.radius) for h in env.humans)
    if env.robot.reached_destination():
        outcome = 'SUCCESS'
    elif collision:
        outcome = 'COLLISION'
    else:
        outcome = 'TIMEOUT'

    print(f"Case {case_id} (Alpha={alpha}, H={horizon}s, margin={margin}m) -> {outcome} in {step} steps! Net Progress: {init_dist - final_dist:.2f}m")
    return outcome

if __name__ == '__main__':
    run_test(90000, alpha=0.15, horizon=1.0, margin=0.10)
