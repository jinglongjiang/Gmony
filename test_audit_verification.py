#!/usr/bin/env python3
"""Rigorous verification: Test exact effect of set_phase('test') and module ablations."""

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

def run_arm(arm_name, set_phase_test=True, progress_alpha=0.0, use_acps=False, use_svo=False, num_cases=20):
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
    net.progress_reward = progress_alpha

    if set_phase_test:
        net.set_phase('test')
        # Check standard eval params
        net.test_min_clearance = 0.2
        net.test_risk_lambda = 0.8
        net.test_action_smoothing = 0.3
    else:
        # What happened in Cycle 4
        net._phase = 'train'
        net.test_min_clearance = 0.0
        net.test_risk_lambda = 0.0
        net.test_action_smoothing = 0.0

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

    results = {'success': 0, 'collision': 0, 'timeout': 0, 'net_progress': []}

    for case_idx in range(num_cases):
        case_id = 90000 + case_idx
        env.case_counter['test'] = case_id
        obs, _ = env.reset(options={'test_case': case_id})
        net.reset_episode_stats()
        done = False
        step = 0
        init_dist = np.hypot(robot.px - robot.gx, robot.py - robot.gy)

        while not done and step < 100:
            step += 1
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            action = net.predict(state)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        final_dist = np.hypot(robot.px - robot.gx, robot.py - robot.gy)
        results['net_progress'].append(init_dist - final_dist)

        collision = any(np.linalg.norm([env.robot.px - h.px, env.robot.py - h.py]) < (env.robot.radius + h.radius) for h in env.humans)
        if env.robot.reached_destination():
            outcome = 'success'
        elif collision:
            outcome = 'collision'
        else:
            outcome = 'timeout'
        results[outcome] += 1

    sr = 100.0 * results['success'] / num_cases
    cr = 100.0 * results['collision'] / num_cases
    tr = 100.0 * results['timeout'] / num_cases
    avg_prog = np.mean(results['net_progress'])
    print(f"[{arm_name:<30}]: SR={sr:5.1f}%, CR={cr:5.1f}%, TR={tr:5.1f}%, NetProg={avg_prog:6.2f}m")
    return sr, cr, tr

if __name__ == '__main__':
    print("Evaluating 20 cases of 20S across configurations:")
    # Arm 1: Cycle 4 Baseline (without set_phase('test'), progress=0.0)
    run_arm("Cycle4 Flawed Baseline (_phase=train)", set_phase_test=False, progress_alpha=0.0, num_cases=20)

    # Arm 2: Real Standard Protocol Baseline (with set_phase('test'), margin=0.2, risk=0.8, smooth=0.3)
    run_arm("Standard Protocol Baseline", set_phase_test=True, progress_alpha=0.0, num_cases=20)

    # Arm 3: Standard Protocol + Progress Alpha=0.15 (DPP alone on standard baseline)
    run_arm("Standard + DPP (alpha=0.15)", set_phase_test=True, progress_alpha=0.15, num_cases=20)
