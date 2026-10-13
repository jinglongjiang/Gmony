#!/usr/bin/env python3
"""Diagnostic: Inspect why baseline policy collides and freezes in 20S.
Analyze the chosen action's continuous-time trajectory and TTC relative to all 20 humans.
"""

import sys
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

def compute_action_ttc(rob_pos, rob_radius, cmd_vel, humans, horizon=3.0):
    """Compute exact continuous-time TTC and min clearance for a candidate action (vx, vy)
    over time t in [0, horizon], assuming constant pedestrian velocity.
    """
    vx, vy = cmd_vel
    min_ttc = float('inf')
    min_clr = float('inf')
    constraint_human = None

    for idx, h in enumerate(humans):
        # Relative position and velocity
        rx = h.px - rob_pos[0]
        ry = h.py - rob_pos[1]
        rvx = h.vx - vx
        rvy = h.vy - vy

        comb_r = rob_radius + h.radius

        # Distance squared: ||r + t * rv||^2 = ||r||^2 + 2*t*(r . rv) + t^2 * ||rv||^2
        r_dot_rv = rx * rvx + ry * rvy
        rv_sq = rvx * rvx + rvy * rvy

        # Initial distance
        d0 = np.hypot(rx, ry) - comb_r
        if d0 < min_clr:
            min_clr = d0

        if rv_sq < 1e-8:
            # Relative speed is near zero
            continue

        # Time of closest approach
        t_cpa = -r_dot_rv / rv_sq
        if t_cpa > 0:
            t_eval = min(t_cpa, horizon)
            d_cpa = np.hypot(rx + t_eval * rvx, ry + t_eval * rvy) - comb_r
            if d_cpa < min_clr:
                min_clr = d_cpa

            # Check if there is collision (distance < comb_r, i.e., clearance < 0)
            disc = r_dot_rv * r_dot_rv - rv_sq * (rx * rx + ry * ry - comb_r * comb_r)
            if disc >= 0:
                # Collision occurs!
                t_col = (-r_dot_rv - np.sqrt(disc)) / rv_sq
                if 0 <= t_col <= horizon and t_col < min_ttc:
                    min_ttc = t_col
                    constraint_human = idx

    return min_ttc, min_clr, constraint_human

def run_diagnostic(num_cases=20):
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

    collision_ttc_stats = []
    step_ttc_stats = []
    blind_endpoint_cases = 0 # Actions safe at t=0.25s endpoint but colliding within 1.0s

    for case_idx in range(num_cases):
        case_id = 90000 + case_idx
        env.case_counter['test'] = case_id
        obs, _ = env.reset(options={'test_case': case_id})
        net.reset_episode_stats()
        done = False
        step = 0

        while not done and step < 100:
            step += 1
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            rob_pos = (robot.px, robot.py)

            action = net.predict(state)
            cmd = (action.vx, action.vy)

            ttc, min_clr, h_idx = compute_action_ttc(rob_pos, robot.radius, cmd, env.humans, horizon=2.0)
            step_ttc_stats.append(ttc)

            # Check endpoint clearance at t=0.25s
            ep_x = robot.px + cmd[0] * 0.25
            ep_y = robot.py + cmd[1] * 0.25
            ep_clr = min(np.hypot(ep_x - (h.px + h.vx * 0.25), ep_y + (h.py + h.vy * 0.25)) - (robot.radius + h.radius) for h in env.humans)

            if ep_clr >= 0.20 and ttc < 1.0:
                blind_endpoint_cases += 1

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Check if this step collided
            is_col = any(np.hypot(robot.px - h.px, robot.py - h.py) < (robot.radius + h.radius) for h in env.humans)
            if is_col:
                collision_ttc_stats.append((case_id, step, ttc, min_clr))

    print(f"\n=== Diagnostic Summary over {num_cases} episodes of 20S ===")
    print(f"Total steps simulated: {len(step_ttc_stats)}")
    print(f"Endpoint Blindness instances (Safe at t=0.25s [>=0.2m], but colliding within 1.0s): {blind_endpoint_cases}")
    print(f"Actual Collisions occurred: {len(collision_ttc_stats)}")
    for c_id, st, ttc, clr in collision_ttc_stats:
        print(f"  Case {c_id} step {st}: Action TTC={ttc:.2f}s, min_clr={clr:.3f}m")

if __name__ == '__main__':
    run_diagnostic()
