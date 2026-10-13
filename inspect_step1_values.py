#!/usr/bin/env python3
"""Inspect candidate action scores at step 1 of Case 90000."""

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
from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized

def inspect_step1(case_id=90000):
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

    env.case_counter['test'] = case_id
    obs, _ = env.reset(options={'test_case': case_id})
    net.reset_episode_stats()

    state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
    
    current_34 = net._build_joint_state_34(state.self_state, state.human_states)
    current_token = _batch_joint34_to_tokens_vectorized(current_34.reshape(1, -1))[0]

    commands, cur_tok, next_states_tensor, rew_batch, dmins_batch = net._score_candidates_vectorized(state, current_token)
    
    with torch.no_grad():
        next_values = net.forward_candidate_values(next_states_tensor)

    rewards_tensor = torch.tensor(rew_batch.tolist(), device=net.device)
    total_values = net.candidate_action_values(state, commands, rewards_tensor, next_values)

    print(f"Goal: ({robot.gx}, {robot.gy}), Robot: ({robot.px}, {robot.py})")
    print(f"Top 10 highest total_values:")
    indices = torch.argsort(total_values, descending=True)[:10]
    for idx in indices:
        cmd = commands[idx.item()]
        r = rew_batch[idx.item()]
        v = next_values[idx.item()].item()
        tot = total_values[idx.item()].item()
        dmin = dmins_batch[idx.item()]
        print(f"  Action [{idx.item():2d}]: vx={cmd.vx:5.2f}, vy={cmd.vy:5.2f} | Total={tot:8.4f} (Rew={r:6.3f}, Val={v:6.3f}) | dmin={dmin:5.2f}m")

    print(f"\nForward actions (vy > 0.5):")
    forward_indices = [i for i, c in enumerate(commands) if c.vy > 0.5]
    for idx in forward_indices[:5]:
        cmd = commands[idx]
        r = rew_batch[idx]
        v = next_values[idx].item()
        tot = total_values[idx].item()
        dmin = dmins_batch[idx]
        print(f"  Action [{idx:2d}]: vx={cmd.vx:5.2f}, vy={cmd.vy:5.2f} | Total={tot:8.4f} (Rew={r:6.3f}, Val={v:6.3f}) | dmin={dmin:5.2f}m")

if __name__ == '__main__':
    inspect_step1(90000)
