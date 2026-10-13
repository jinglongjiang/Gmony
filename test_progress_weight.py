#!/usr/bin/env python3
"""Test the effect of Goal Progress Weight in candidate evaluation on Case 90000."""

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

def test_progress_weight(alpha, case_id=90000):
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
        action = net.predict(state)
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

    print(f"Alpha={alpha:4.2f} | Outcome: {outcome:<9} | Steps: {step:3d} | Net Progress: {init_dist - final_dist:6.2f}m | Final Pos: ({robot.px:5.2f}, {robot.py:5.2f})")

if __name__ == '__main__':
    for a in [0.0, 0.05, 0.1, 0.2, 0.5]:
        test_progress_weight(a, 90000)
