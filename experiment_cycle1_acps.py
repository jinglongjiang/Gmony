#!/usr/bin/env python3
"""Cycle 1 Experiment: Empirical Verification of Action-Conditioned Pedestrian Selection (ACPS)

Hypothesis:
Static TTC5 selects pedestrians using current robot velocity, causing >40% omission
of the true constraint pedestrian for alternative candidate actions.
Action-Conditioned Pedestrian Selection (ACPS) conditions human selection on candidate velocity,
drastically reducing omission of the true constraint pedestrian without expanding token size (K=5).
"""

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
from crowd_sim.envs.utils.action import ActionXY

DT = 0.25

def select_static_ttc5(robot, humans):
    """Original static TTC5 selection using current robot velocity."""
    if len(humans) <= 5:
        return list(range(len(humans)))
    h = np.asarray([[x.px, x.py, x.vx, x.vy] for x in humans], dtype=np.float64)
    rp = np.asarray([robot.px, robot.py, robot.vx, robot.vy], dtype=np.float64)
    dx, dy = h[:, 0] - rp[0], h[:, 1] - rp[1]
    dvx, dvy = h[:, 2] - rp[2], h[:, 3] - rp[3]
    d = np.sqrt(dx * dx + dy * dy + 1e-6)
    closing = -(dx * dvx + dy * dvy) / (d + 1e-6)
    score = np.where(closing > 0.1, d / (closing + 1e-6), d * 10)
    return sorted(np.argsort(score)[:5].tolist())

def select_action_conditioned_top5(robot, humans, candidate_vx, candidate_vy):
    """Action-Conditioned Selection (ACPS): conditions closing speed on candidate velocity."""
    if len(humans) <= 5:
        return list(range(len(humans)))
    h = np.asarray([[x.px, x.py, x.vx, x.vy] for x in humans], dtype=np.float64)
    dx, dy = h[:, 0] - robot.px, h[:, 1] - robot.py
    dvx = h[:, 2] - candidate_vx
    dvy = h[:, 3] - candidate_vy
    d = np.sqrt(dx * dx + dy * dy + 1e-6)
    closing = -(dx * dvx + dy * dvy) / (d + 1e-6)
    # Predicted closest approach distance along ray
    # t_star = clip(-(dx*dvx + dy*dvy) / (dvx^2 + dvy^2), 0, 2.0)
    rel_speed_sq = dvx * dvx + dvy * dvy + 1e-12
    t_star = np.clip(-(dx * dvx + dy * dvy) / rel_speed_sq, 0.0, 2.0)
    min_dist = np.sqrt((dx + t_star * dvx)**2 + (dy + t_star * dvy)**2)
    # Score favors humans with smallest predicted closest approach distance
    score = min_dist + 0.1 * np.where(closing > 0.1, d / (closing + 1e-6), d * 5)
    return sorted(np.argsort(score)[:5].tolist())

def find_ground_truth_constraint_pedestrian(robot, humans, action):
    """Find the pedestrian with the absolute minimum swept clearance under action."""
    vx, vy = action.vx, action.vy
    p0 = np.array([robot.px, robot.py])
    min_clearance = float('inf')
    constraint_idx = None
    for i, h in enumerate(humans):
        # Human position over [0, DT]
        hp0 = np.array([h.px, h.py])
        h_vx, h_vy = h.vx, h.vy
        # Relative motion
        rel_pos = p0 - hp0
        rel_vel = np.array([vx - h_vx, vy - h_vy])
        denom = np.dot(rel_vel, rel_vel)
        if denom > 1e-12:
            tau = np.clip(-np.dot(rel_pos, rel_vel) / denom, 0.0, DT)
        else:
            tau = 0.0
        closest_pt = rel_pos + tau * rel_vel
        dist = np.linalg.norm(closest_pt) - robot.radius - h.radius
        if dist < min_clearance:
            min_clearance = dist
            constraint_idx = i
    return constraint_idx, min_clearance

def run_experiment(num_cases=50):
    print("=" * 70)
    print("CYCLE 1: Action-Conditioned Pedestrian Selection (ACPS) Empirical Audit")
    print("=" * 70)

    # 1. Config & Environment
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

    total_steps = 0
    ttc5_omissions = 0
    acps_omissions = 0
    
    collision_steps = 0
    ttc5_collision_omissions = 0
    acps_collision_omissions = 0

    danger_steps = 0 # clearance < 0.2m
    ttc5_danger_omissions = 0
    acps_danger_omissions = 0

    print(f"Running evaluation on {num_cases} episodes of 20-human square crossing...")

    for case_idx in range(num_cases):
        case_id = 90000 + case_idx
        env.case_counter['test'] = case_id
        obs, _ = env.reset(options={'test_case': case_id})
        net.reset_episode_stats()
        done = False
        step = 0

        while not done and step < 100:
            step += 1
            total_steps += 1
            from crowd_sim.envs.utils.state import JointState
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            action = net.predict(state)

            # Ground truth constraint human for the executed action
            constraint_h, min_clr = find_ground_truth_constraint_pedestrian(
                state.self_state, state.human_states, action
            )

            # 1. Static TTC5 selection
            static_selected = select_static_ttc5(state.self_state, state.human_states)
            ttc5_missed = (constraint_h not in static_selected)
            if ttc5_missed:
                ttc5_omissions += 1

            # 2. Action-Conditioned Selection (ACPS)
            acps_selected = select_action_conditioned_top5(
                state.self_state, state.human_states, action.vx, action.vy
            )
            acps_missed = (constraint_h not in acps_selected)
            if acps_missed:
                acps_omissions += 1

            # Danger analysis
            if min_clr < 0.2:
                danger_steps += 1
                if ttc5_missed:
                    ttc5_danger_omissions += 1
                if acps_missed:
                    acps_danger_omissions += 1

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if info and ('Collision' in str(info) or info.get('event') == 'collision'):
                collision_steps += 1
                if ttc5_missed:
                    ttc5_collision_omissions += 1
                if acps_missed:
                    acps_collision_omissions += 1

        if (case_idx + 1) % 10 == 0:
            print(f"  Processed {case_idx + 1}/{num_cases} episodes...")

    # Results summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY:")
    print(f"Total steps analyzed: {total_steps}")
    print(f"Overall TTC5 Omission Rate: {ttc5_omissions}/{total_steps} ({100.0 * ttc5_omissions / total_steps:.2f}%)")
    print(f"Overall ACPS Omission Rate: {acps_omissions}/{total_steps} ({100.0 * acps_omissions / total_steps:.2f}%)")
    print("-" * 50)
    print(f"Danger steps (clearance < 0.2m): {danger_steps}")
    if danger_steps > 0:
        print(f"Danger TTC5 Omission Rate:  {ttc5_danger_omissions}/{danger_steps} ({100.0 * ttc5_danger_omissions / danger_steps:.2f}%)")
        print(f"Danger ACPS Omission Rate:  {acps_danger_omissions}/{danger_steps} ({100.0 * acps_danger_omissions / danger_steps:.2f}%)")
    print("-" * 50)
    print(f"Collision steps: {collision_steps}")
    if collision_steps > 0:
        print(f"Collision TTC5 Omission Rate: {ttc5_collision_omissions}/{collision_steps} ({100.0 * ttc5_collision_omissions / collision_steps:.2f}%)")
        print(f"Collision ACPS Omission Rate: {acps_collision_omissions}/{collision_steps} ({100.0 * acps_collision_omissions / collision_steps:.2f}%)")
    print("=" * 70)

    # Formal Verdict
    reduction_overall = (ttc5_omissions - acps_omissions) / max(1, ttc5_omissions) * 100.0
    passed = acps_omissions < ttc5_omissions and (acps_danger_omissions < ttc5_danger_omissions)
    print(f"\nOmission Reduction: {reduction_overall:.2f}%")
    print(f"Cycle 1 Verdict: {'PASS (Hypothesis Validated)' if passed else 'FAIL'}")
    return passed

if __name__ == '__main__':
    run_experiment(num_cases=50)
