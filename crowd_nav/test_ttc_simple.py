#!/usr/bin/env python3
"""
Mamba Policy Testing - Run 7 (Optimized)
Strategy:
1. No Patch (Dirty Data) - Proven to give best success (44%).
2. Config: T=10, dt=0.1 (Match default.yaml exactly).
3. v_max=1.3 (Slightly conservative vs 1.5 to reduce collision).
4. Pre-Sort Humans: Fixes blindness in >5 human scenarios (Critical for square_crossing).
"""
import logging
import argparse
import configparser
import os
import sys
import re
import torch
import numpy as np
import random
import time
from tqdm import tqdm
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.contracts import GRID, discrete_index_to_action, joint34_to_tokens

# NO Monkey Patch

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_ttc(robot, human, v_floor=0.1):
    rx, ry, rvx, rvy = robot.px, robot.py, robot.vx, robot.vy
    hx, hy, hvx, hvy = human.px, human.py, human.vx, human.vy
    rel_x, rel_y = hx - rx, hy - ry
    rel_vx, rel_vy = hvx - rvx, hvy - rvy
    dist = np.hypot(rel_x, rel_y) + 1e-6
    closing = -(rel_x * rel_vx + rel_y * rel_vy) / dist
    # 静止/背离也按距离给出“有效TTC”，避免危险人被排到最后
    eff = closing if closing > v_floor else v_floor
    return dist / (eff + 1e-6)

def _rank_humans(humans, robot, method='ttc', v_floor=0.1):
    if method == 'dist':
        dists = [np.linalg.norm(np.array([h.px, h.py]) - np.array([robot.px, robot.py])) for h in humans]
        return [h for _, h in sorted(zip(dists, humans), key=lambda pair: pair[0])]

    scored = []
    for h in humans:
        dist = np.hypot(h.px - robot.px, h.py - robot.py)
        ttc = compute_ttc(robot, h, v_floor=v_floor)
        scored.append((ttc, dist, h))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [h for _, _, h in scored]

def _simulate_step(env, action):
    # update=False 仍会改动 env._no_prog_steps，这里手动回滚以避免副作用
    no_prog = getattr(env, '_no_prog_steps', None)
    out = env.step(action, update=False)
    if no_prog is not None:
        env._no_prog_steps = no_prog
    if len(out) == 5:
        _, _, terminated, truncated, info = out
    else:
        _, _, done, info = out
        terminated = bool(done)
        truncated = False
    return terminated, truncated, info

def _is_bad(info):
    if not isinstance(info, dict):
        return False
    event = str(info.get('event', '')).lower()
    return ('collision' in event) or ('timeout' in event) or ('no_progress' in event)

def _score_action(env, action, goal, w_prog=1.0, w_safe=0.6, w_ttc=0.2, min_clearance=0.05):
    term, trunc, info = _simulate_step(env, action)
    if (term or trunc) and _is_bad(info):
        return -1e9

    curr = np.array(env.robot.get_position(), dtype=float)
    dist_curr = np.linalg.norm(curr - goal)
    nx, ny = env.robot.compute_position(action, env.time_step)
    dist_next = np.linalg.norm(np.array([nx, ny], dtype=float) - goal)
    progress = dist_curr - dist_next

    dmin = float(info.get('dmin', info.get('min_dist', 0.0))) if isinstance(info, dict) else 0.0
    ttc = float(info.get('ttc', np.inf)) if isinstance(info, dict) else np.inf
    safe = np.clip(dmin, -0.5, 1.5)
    ttc_score = 0.0 if not np.isfinite(ttc) else min(ttc, 5.0) / 5.0

    speed = np.hypot(action.vx, action.vy)
    slow_pen = -0.03 if speed < 0.05 else 0.0
    clearance_pen = -2.0 * max(0.0, min_clearance - dmin)

    return w_prog * progress + w_safe * safe + w_ttc * ttc_score + slow_pen + clearance_pen

def _candidate_actions(robot, action_rl, action_orca=None, n_angles=7, speed_scales=(1.0, 0.7)):
    candidates = []
    dx, dy = robot.gx - robot.px, robot.gy - robot.py
    base = np.arctan2(dy, dx)

    # 以目标方向为中心的扇形采样
    offsets = [0.0, np.deg2rad(30), -np.deg2rad(30), np.deg2rad(60), -np.deg2rad(60), np.deg2rad(90), -np.deg2rad(90)]
    offsets = offsets[:max(1, min(len(offsets), n_angles))]

    for s in speed_scales:
        speed = max(0.0, robot.v_pref * float(s))
        for off in offsets:
            ang = base + off
            candidates.append(ActionXY(speed * np.cos(ang), speed * np.sin(ang)))

    if action_rl is not None:
        candidates.append(action_rl)
    if action_orca is not None:
        candidates.append(action_orca)

    # 去重（避免重复评估）
    uniq = []
    seen = set()
    for a in candidates:
        key = (round(a.vx, 2), round(a.vy, 2))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)
    return uniq

def _choose_action_mpc(env, candidates, w_prog=1.0, w_safe=0.6, w_ttc=0.2, min_clearance=0.05):
    goal = np.array(env.robot.get_goal_position(), dtype=float)
    best = None
    best_score = -1e9
    for a in candidates:
        score = _score_action(env, a, goal, w_prog=w_prog, w_safe=w_safe, w_ttc=w_ttc, min_clearance=min_clearance)
        if score > best_score:
            best_score = score
            best = a
    return best if best is not None else candidates[0]

def test_episode(
    env, robot, policy, test_case_idx,
    robot_start=None, robot_goal=None,
    human_sort='ttc', ttc_v_floor=0.1,
    mpc_lite=False, mpc_angles=7, mpc_speed_scales=(1.0, 0.7),
    mpc_w_prog=1.0, mpc_w_safe=0.6, mpc_w_ttc=0.2, mpc_min_clearance=0.05
):
    if hasattr(policy, 'reset_episode_stats'):
        policy.reset_episode_stats()

    reset_result = env.reset(seed=test_case_idx)
    if isinstance(reset_result, tuple):
        ob = reset_result[0]
    else:
        ob = reset_result

    if robot_start is not None:
        robot.px, robot.py = robot_start
        robot.vx, robot.vy = 0.0, 0.0
    if robot_goal is not None:
        robot.gx, robot.gy = robot_goal

    done = False
    steps = 0
    robot_positions = [robot.get_position()]
    min_dists = []
    
    max_steps = 500

    while not done and steps < max_steps:
        # --- Manual Act with Pre-Sorting (Fixes 10-human blindness) ---
        # 1. Get full state
        robot_state = robot.get_full_state()
        
        # 2. Get sorted humans (risk-aware)
        humans = env.humans
        sorted_humans = _rank_humans(humans, robot, method=human_sort, v_floor=ttc_v_floor)
        
        # 3. Construct JointState
        # 只取最危险的5人喂给模型（保持34D输入）
        human_states = [h.get_observable_state() for h in sorted_humans[:5]]
        state = JointState(robot_state, human_states)
        
        # 4. Predict
        action = policy.predict(state)
        if mpc_lite:
            candidates = _candidate_actions(
                robot, action, action_orca=None,
                n_angles=mpc_angles, speed_scales=mpc_speed_scales
            )
            action = _choose_action_mpc(
                env, candidates,
                w_prog=mpc_w_prog, w_safe=mpc_w_safe, w_ttc=mpc_w_ttc,
                min_clearance=mpc_min_clearance
            )

        # --------------------------------------------------------------

        step_result = env.step(action)
        if len(step_result) == 5:
            ob, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            ob, reward, done, info = step_result
        robot_positions.append(robot.get_position())

        min_dist = float('inf')
        for human in env.humans:
            dist = np.linalg.norm(np.array([robot.px, robot.py]) - np.array([human.px, human.py]))
            dist -= (robot.radius + human.radius)
            min_dist = min(min_dist, dist)
        min_dists.append(min_dist)
        steps += 1

    outcome = 'timeout'
    if isinstance(info, dict):
        event = info.get('event', '').lower()
        if any(x in event for x in ['reach_goal', 'success', 'reachgoal']):
            outcome = 'success'
        elif any(x in event for x in ['collision']):
            outcome = 'collision'
    else:
        event_name = info.__class__.__name__
        if event_name == 'ReachGoal':
            outcome = 'success'
        elif event_name == 'Collision':
            outcome = 'collision'

    return outcome, steps, robot_positions, min_dists

def compute_metrics(outcome, steps, robot_positions, min_dists, start_pos, goal_pos, time_step):
    metrics = {}
    metrics['time'] = steps * time_step
    if len(robot_positions) > 1:
        path_length = sum(np.linalg.norm(np.array(robot_positions[i+1]) - np.array(robot_positions[i]))
                         for i in range(len(robot_positions)-1))
        optimal_length = np.linalg.norm(np.array(goal_pos) - np.array(start_pos))
        metrics['path_efficiency'] = optimal_length / max(path_length, 1e-6) if path_length > 0 else 0.0
    else:
        metrics['path_efficiency'] = 0.0
    metrics['min_separation'] = np.min(min_dists) if min_dists else float('inf')
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='mamba', help='Policy name')
    parser.add_argument('--model_dir', type=str, default='/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav/runs/mamba_vl', help='Model directory')
    parser.add_argument('--weights', type=str, default='rl_model_ep10000_1.pth', help='Checkpoint name or "latest"')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--episodes', type=int, default=500, help='Episodes per scenario')
    parser.add_argument('--test_case', type=int, default=None, help='Run only specified test case')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--human_sort', type=str, default='ttc', choices=['ttc', 'dist'],
                        help='human selection: ttc (risk) or dist (nearest)')
    parser.add_argument('--ttc_v_floor', type=float, default=0.1,
                        help='effective closing speed floor for TTC ranking')
    parser.add_argument('--mpc_lite', action='store_true',
                        help='enable MPC-lite action selection (test-time wrapper)')
    parser.add_argument('--mpc_angles', type=int, default=7,
                        help='number of goal-relative angle samples')
    parser.add_argument('--mpc_speed_scales', type=str, default='1.0,0.7',
                        help='comma-separated speed scales for candidate actions')
    parser.add_argument('--mpc_w_prog', type=float, default=1.0,
                        help='MPC-lite weight for goal progress')
    parser.add_argument('--mpc_w_safe', type=float, default=0.6,
                        help='MPC-lite weight for clearance')
    parser.add_argument('--mpc_w_ttc', type=float, default=0.2,
                        help='MPC-lite weight for TTC')
    parser.add_argument('--mpc_min_clearance', type=float, default=0.05,
                        help='clearance penalty threshold')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    print("[INFO] Config: T=12, dt=0.25, v_max=1.0 (MATCHED TO TRAINING)")
    print("[INFO] Feature: Pre-Sorting Humans enabled (Fixes >5 human blindness)")

    # parse MPC speed scales
    try:
        mpc_speed_scales = tuple(
            float(s.strip()) for s in args.mpc_speed_scales.split(',') if s.strip()
        )
    except ValueError:
        mpc_speed_scales = (1.0, 0.7)
    if not mpc_speed_scales:
        mpc_speed_scales = (1.0, 0.7)

    print(f"[INFO] Human sort: {args.human_sort} (v_floor={args.ttc_v_floor})")
    print(f"[INFO] MPC-lite: {'ON' if args.mpc_lite else 'OFF'} | angles={args.mpc_angles} | scales={mpc_speed_scales}")

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    test_cases = [
        {'desc': 'baseline_circle', 'sim': 'circle_crossing', 'human_num': 5,  'circle_radius': 4.0},
        {'desc': 'dense_circle',    'sim': 'circle_crossing', 'human_num': 10, 'circle_radius': 4.0},
        {'desc': 'large_circle',    'sim': 'circle_crossing', 'human_num': 12, 'circle_radius': 6.0},
        {'desc': 'baseline_square', 'sim': 'square_crossing', 'human_num': 10, 'square_width': 10.0},
        {'desc': 'dense_square',    'sim': 'square_crossing', 'human_num': 15, 'square_width': 10.0},
        {'desc': 'large_square',    'sim': 'square_crossing', 'human_num': 20, 'square_width': 15.0},
    ]
    
    if args.test_case is not None:
        if 0 <= args.test_case < len(test_cases):
            test_cases = [test_cases[args.test_case]]

    print(f"Using device: {device}")
    
    base_seed = int(time.time() * 1000) % (2**31)
    random.seed(base_seed)
    np.random.seed(base_seed)

    for case_idx, case in enumerate(test_cases):
        print(f"\n{'='*70}")
        print(f"Test Case [{case_idx}]: {case['desc']} | {case['human_num']} humans")
        print(f"{'='*70}")

        env_config = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
        env_config.read(args.env_config, encoding='utf-8')
        policy_config = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
        policy_config.read(args.policy_config, encoding='utf-8')
        
        # [FIX 1] Config Injection - MATCH TRAINING CONFIG
        if not policy_config.has_section('buffer'): policy_config.add_section('buffer')
        policy_config.set('buffer', 'seq_len', '12')  # Match training: T=12 (from train.config)

        if not policy_config.has_section('robot'): policy_config.add_section('robot')
        policy_config.set('robot', 'v_pref', '1.0')   # Match training

        # [FIX 2] Env Injection - MATCH TRAINING CONFIG
        if not env_config.has_section('env'): env_config.add_section('env')
        env_config.set('env', 'time_step', '0.25')  # Match training: dt=0.25
        env_config.set('env', 'time_limit', '100')

        if not env_config.has_section('robot'): env_config.add_section('robot')
        env_config.set('robot', 'v_pref', '1.0')

        for section in policy_config.sections():
            if not env_config.has_section(section): env_config.add_section(section)
            for key, value in policy_config.items(section): env_config.set(section, key, value)

        if not env_config.has_section('sim'): env_config.add_section('sim')
        env_config.set('sim', 'test_sim', case['sim'])
        env_config.set('sim', 'human_num', str(case['human_num']))
        if 'circle_radius' in case: env_config.set('sim', 'circle_radius', str(case['circle_radius']))
        if 'square_width' in case: env_config.set('sim', 'square_width', str(case['square_width']))

        try:
            policy_class = policy_factory[args.policy]
            policy = policy_class(policy_config)
        except:
            policy = policy_class()

        if args.weights == 'latest':
            files = [f for f in os.listdir(args.model_dir) if re.match(r'rl_model_ep\d+\.pth', f)]
            if files:
                files.sort(key=lambda x: os.path.getmtime(os.path.join(args.model_dir, x)))
                weights_path = os.path.join(args.model_dir, files[-1])
            else:
                weights_path = os.path.join(args.model_dir, 'rl_model.pth')
        else:
            weights_path = os.path.join(args.model_dir, args.weights)

        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('policy_state', checkpoint.get('value_state', checkpoint.get('model', checkpoint)))
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        if hasattr(policy, 'load_state_dict'): policy.load_state_dict(state_dict)
        elif hasattr(policy, 'model'): policy.model.load_state_dict(state_dict)
        
        if hasattr(policy, 'to'): policy.to(device)
        if hasattr(policy, 'eval'): policy.eval()
        elif hasattr(policy, 'model'): policy.model.eval()

        env = CrowdSim()
        env.configure(env_config)
        env.phase = 'test'
        robot = Robot(env_config, 'robot')
        robot.set_policy(policy)
        robot.env = env
        env.set_robot(robot)

        if hasattr(policy, 'set_env'): policy.set_env(env)
        if hasattr(policy, 'set_phase'): policy.set_phase('test')
        if hasattr(policy, 'set_env_dt'): policy.set_env_dt(0.25)  # Match training dt

        success_count = 0
        collision_count = 0
        timeout_count = 0
        
        pbar = tqdm(range(args.episodes), ncols=100)
        for ep in pbar:
            episode_seed = random.randint(0, 2**31 - 1)
            
            outcome, steps, robot_positions, min_dists = test_episode(
                env, robot, policy, episode_seed,
                robot_start=case.get('robot_start'),
                robot_goal=case.get('robot_goal'),
                human_sort=args.human_sort, ttc_v_floor=args.ttc_v_floor,
                mpc_lite=args.mpc_lite, mpc_angles=args.mpc_angles, mpc_speed_scales=mpc_speed_scales,
                mpc_w_prog=args.mpc_w_prog, mpc_w_safe=args.mpc_w_safe, mpc_w_ttc=args.mpc_w_ttc,
                mpc_min_clearance=args.mpc_min_clearance
            )

            if outcome == 'success': success_count += 1
            elif outcome == 'collision': collision_count += 1
            else: timeout_count += 1
            
            pbar.set_postfix({'S': f'{success_count}/{ep+1}', 'C': f'{collision_count}/{ep+1}'})

        print(f"\nResults for {case['desc']}:")
        print(f"  SUCCESS:   {success_count}/{args.episodes} ({100*success_count/args.episodes:.1f}%)")
        print(f"  COLLISION: {collision_count}/{args.episodes} ({100*collision_count/args.episodes:.1f}%)")
        print(f"  TIMEOUT:   {timeout_count}/{args.episodes} ({100*timeout_count/args.episodes:.1f}%)")

if __name__ == '__main__':
    main()
