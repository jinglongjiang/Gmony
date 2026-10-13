#!/usr/bin/env python3
"""Establish true ground-truth baseline on standard 20S (square_width=10, set_phase('test'))."""

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

def benchmark_standard_baseline(num_cases=50):
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
    net.set_phase('test')
    net.test_min_clearance = 0.2
    net.test_risk_lambda = 0.8
    net.test_action_smoothing = 0.3

    env = CrowdSim()
    env.configure(env_config)
    env.human_num = 20
    env.square_width = 10.0  # Standard KDA square_width!
    env.circle_radius = 4.0
    env.phase = 'test'
    env.test_sim = 'square_crossing'

    robot = Robot(env_config, 'robot')
    robot.set_policy(net)
    robot.env = env
    env.set_robot(robot)

    results = {'success': 0, 'collision': 0, 'timeout': 0, 'times': [], 'case_details': {}}

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
            action = net.predict(state)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        collision = any(np.linalg.norm([env.robot.px - h.px, env.robot.py - h.py]) < (env.robot.radius + h.radius) for h in env.humans)
        if env.robot.reached_destination():
            outcome = 'SUCCESS'
            results['times'].append(env.global_time)
        elif collision:
            outcome = 'COLLISION'
        else:
            outcome = 'TIMEOUT'
        results[outcome.lower()] += 1
        results['case_details'][case_id] = outcome

    sr = 100.0 * results['success'] / num_cases
    cr = 100.0 * results['collision'] / num_cases
    tr = 100.0 * results['timeout'] / num_cases
    mean_time = np.mean(results['times']) if len(results['times']) else 0.0

    print(f"\n=======================================================")
    print(f"True Standard 20S Baseline (square_width=10.0, 50 cases):")
    print(f"SR = {sr:.1f}% ({results['success']}/{num_cases})")
    print(f"CR = {cr:.1f}% ({results['collision']}/{num_cases})")
    print(f"TR = {tr:.1f}% ({results['timeout']}/{num_cases})")
    print(f"Mean Navigation Time (Success): {mean_time:.2f}s")
    print(f"=======================================================\n")
    return results

if __name__ == '__main__':
    benchmark_standard_baseline(50)
