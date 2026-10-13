#!/usr/bin/env python3
"""
Mamba-PPO Policy Testing
Based on test.py but uses PPO-trained model architecture (mamba_rl_ppo.py)
"""
import logging
import argparse
import configparser
import csv
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

from crowd_nav.policy.mamba_rl_ppo import MambaRL as MambaRL_PPO
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.contracts import GRID, action_to_discrete_index


OSCILLATION_FIELDS = [
    'run_label',
    'scenario',
    'scenario_index',
    'episode',
    'seed',
    'outcome',
    'steps',
    'duration_s',
    'path_length_m',
    'action_switch_count',
    'action_switch_hz',
    'turn_reversal_count',
    'turn_reversal_hz',
    'heading_flip_count',
    'heading_flip_hz',
    'total_abs_turn_rad',
    'curvature_rad_per_m',
    'lateral_sign_change_count',
    'lateral_sign_change_hz',
    'stalled_window_ratio',
    'oscillatory_stall_window_ratio',
    'backtracking_ratio',
    'goal_progress_efficiency',
]

# NO Monkey Patch

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _sign_change_count(values, deadband):
    signs = []
    for value in values:
        if abs(value) <= deadband:
            continue
        signs.append(1 if value > 0 else -1)
    return sum(curr != prev for prev, curr in zip(signs, signs[1:]))


def compute_oscillation_metrics(actions, positions, start_pos, goal_pos, time_step):
    action_array = np.asarray(actions, dtype=np.float64)
    position_array = np.asarray(positions, dtype=np.float64)
    steps = int(len(action_array))
    duration = steps * float(time_step)

    if steps == 0:
        return {
            'action_switch_count': 0,
            'action_switch_hz': 0.0,
            'turn_reversal_count': 0,
            'turn_reversal_hz': 0.0,
            'heading_flip_count': 0,
            'heading_flip_hz': 0.0,
            'total_abs_turn_rad': 0.0,
            'curvature_rad_per_m': 0.0,
            'lateral_sign_change_count': 0,
            'lateral_sign_change_hz': 0.0,
            'path_length_m': 0.0,
            'stalled_window_ratio': 0.0,
            'oscillatory_stall_window_ratio': 0.0,
            'backtracking_ratio': 0.0,
            'goal_progress_efficiency': 0.0,
        }

    # Quantize continuous PPO actions onto the same 80-action grid used by
    # Mamba-VL before counting action identity changes.
    action_indices = [
        int(action_to_discrete_index(float(vx), float(vy), grid=GRID))
        for vx, vy in action_array
    ]
    action_switch_count = sum(
        curr != prev for prev, curr in zip(action_indices, action_indices[1:])
    )

    speeds = np.linalg.norm(action_array, axis=1)
    headings = np.arctan2(action_array[:, 1], action_array[:, 0])
    heading_deltas = np.array(
        [_wrap_angle(curr - prev) for prev, curr in zip(headings, headings[1:])],
        dtype=np.float64,
    )
    moving_pairs = (speeds[:-1] > 0.05) & (speeds[1:] > 0.05)
    heading_deltas = heading_deltas[moving_pairs]
    turn_reversal_count = _sign_change_count(
        heading_deltas,
        deadband=np.deg2rad(5.0),
    )
    heading_flip_count = int(np.sum(np.abs(heading_deltas) >= (np.pi / 2)))
    total_abs_turn = float(np.sum(np.abs(heading_deltas)))

    if len(position_array) > 1:
        path_length = float(
            np.sum(np.linalg.norm(np.diff(position_array, axis=0), axis=1))
        )
    else:
        path_length = 0.0

    goal_vector = np.asarray(goal_pos, dtype=np.float64) - np.asarray(
        start_pos, dtype=np.float64
    )
    goal_norm = float(np.linalg.norm(goal_vector))
    if goal_norm > 1e-8:
        goal_unit = goal_vector / goal_norm
        lateral_axis = np.array([-goal_unit[1], goal_unit[0]], dtype=np.float64)
        lateral_velocities = action_array @ lateral_axis
        lateral_sign_change_count = _sign_change_count(
            lateral_velocities,
            deadband=0.05,
        )
    else:
        goal_unit = np.zeros(2, dtype=np.float64)
        lateral_sign_change_count = 0
        lateral_velocities = np.zeros(steps, dtype=np.float64)

    if len(position_array) > 1 and goal_norm > 1e-8:
        displacements = np.diff(position_array, axis=0)
        forward_progress = displacements @ goal_unit
        backtracking_ratio = float(np.mean(forward_progress < -0.01))
        net_goal_progress = max(
            0.0,
            goal_norm - float(np.linalg.norm(np.asarray(goal_pos) - position_array[-1])),
        )
        goal_progress_efficiency = net_goal_progress / max(path_length, 1e-8)

        window_steps = max(2, int(round(2.0 / float(time_step))))
        stalled_windows = 0
        oscillatory_stall_windows = 0
        total_windows = max(0, steps - window_steps + 1)
        for start in range(total_windows):
            end = start + window_steps
            window_progress = float(np.sum(forward_progress[start:end]))
            stalled = window_progress < 0.2
            if not stalled:
                continue
            stalled_windows += 1

            window_lateral_changes = _sign_change_count(
                lateral_velocities[start:end],
                deadband=0.05,
            )
            window_actions = action_array[start:end]
            window_speeds = np.linalg.norm(window_actions, axis=1)
            window_headings = np.arctan2(window_actions[:, 1], window_actions[:, 0])
            window_heading_deltas = np.array(
                [
                    _wrap_angle(curr - prev)
                    for prev, curr in zip(window_headings, window_headings[1:])
                ],
                dtype=np.float64,
            )
            window_moving_pairs = (
                (window_speeds[:-1] > 0.05) & (window_speeds[1:] > 0.05)
            )
            window_turn_changes = _sign_change_count(
                window_heading_deltas[window_moving_pairs],
                deadband=np.deg2rad(5.0),
            )
            if window_lateral_changes >= 2 or window_turn_changes >= 2:
                oscillatory_stall_windows += 1

        stalled_window_ratio = (
            stalled_windows / total_windows if total_windows else 0.0
        )
        oscillatory_stall_window_ratio = (
            oscillatory_stall_windows / total_windows if total_windows else 0.0
        )
    else:
        backtracking_ratio = 0.0
        goal_progress_efficiency = 0.0
        stalled_window_ratio = 0.0
        oscillatory_stall_window_ratio = 0.0

    duration_safe = max(duration, 1e-8)
    return {
        'action_switch_count': action_switch_count,
        'action_switch_hz': action_switch_count / duration_safe,
        'turn_reversal_count': turn_reversal_count,
        'turn_reversal_hz': turn_reversal_count / duration_safe,
        'heading_flip_count': heading_flip_count,
        'heading_flip_hz': heading_flip_count / duration_safe,
        'total_abs_turn_rad': total_abs_turn,
        'curvature_rad_per_m': total_abs_turn / max(path_length, 1e-8),
        'lateral_sign_change_count': lateral_sign_change_count,
        'lateral_sign_change_hz': lateral_sign_change_count / duration_safe,
        'path_length_m': path_length,
        'stalled_window_ratio': stalled_window_ratio,
        'oscillatory_stall_window_ratio': oscillatory_stall_window_ratio,
        'backtracking_ratio': backtracking_ratio,
        'goal_progress_efficiency': goal_progress_efficiency,
    }


def test_episode(env, robot, policy, test_case_idx, robot_start=None, robot_goal=None):
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
    executed_actions = []
    start_pos = tuple(float(x) for x in robot.get_position())
    goal_pos = (float(robot.gx), float(robot.gy))
    
    max_steps = 500

    while not done and steps < max_steps:
        # --- Manual Act with the same TTC sorting used by Mamba-VL ---
        # 1. Get full state
        robot_state = robot.get_full_state()
        
        # 2. Sort humans by time to collision (most urgent first)
        humans = env.humans
        ttc_list = []
        for human in humans:
            rel_x = human.px - robot.px
            rel_y = human.py - robot.py
            rel_vx = human.vx - robot.vx
            rel_vy = human.vy - robot.vy
            dist = np.sqrt(rel_x**2 + rel_y**2 + 1e-6)
            closing = -(rel_x * rel_vx + rel_y * rel_vy) / (dist + 1e-6)
            ttc = dist / (closing + 1e-6) if closing > 0.1 else dist * 10
            ttc_list.append((ttc, human))
        sorted_humans = [
            human for _, human in sorted(ttc_list, key=lambda item: item[0])
        ]
        
        # 3. Construct JointState
        human_states = [h.get_observable_state() for h in sorted_humans]
        state = JointState(robot_state, human_states)
        
        # 4. Predict
        action = policy.predict(state)
        executed_actions.append((float(action.vx), float(action.vy)))
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

    oscillation_metrics = compute_oscillation_metrics(
        executed_actions,
        robot_positions,
        start_pos,
        goal_pos,
        env.time_step,
    )
    return outcome, steps, robot_positions, min_dists, oscillation_metrics

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
    parser.add_argument('--policy', type=str, default='mamba_ppo', help='Policy name (mamba_ppo)')
    parser.add_argument('--model_dir', type=str, default='runs', help='Model directory')
    parser.add_argument('--weights', type=str, default='rl_model_ep10000.pth', help='Checkpoint name')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy_ppo.config')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--episodes', type=int, default=500, help='Episodes per scenario')
    parser.add_argument('--test_case', type=int, default=None, help='Run only specified test case')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--oscillation_csv', type=str, default=None,
                        help='Write one row per episode with oscillation metrics')
    parser.add_argument('--run_label', type=str, default='mamba_ppo',
                        help='Label stored in --oscillation_csv')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    print("[INFO] Mamba-PPO Testing")
    print("[INFO] Feature: TTC-Sorting enabled (matched to Mamba-VL)")

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    test_cases = [
        {'case_id': 0, 'desc': 'baseline_circle', 'sim': 'circle_crossing', 'human_num': 5,  'circle_radius': 4.0},
        {'case_id': 1, 'desc': 'baseline_square', 'sim': 'square_crossing', 'human_num': 10, 'square_width': 10.0},
        {'case_id': 2, 'desc': 'dense_circle',    'sim': 'circle_crossing', 'human_num': 10, 'circle_radius': 4.0},
        {'case_id': 3, 'desc': 'dense_square',    'sim': 'square_crossing', 'human_num': 20, 'square_width': 10.0},
        {'case_id': 4, 'desc': 'large_circle',    'sim': 'circle_crossing', 'human_num': 12, 'circle_radius': 6.0},
        {'case_id': 5, 'desc': 'large_square',    'sim': 'square_crossing', 'human_num': 20, 'square_width': 14.0},
    ]
    
    if args.test_case is not None:
        if 0 <= args.test_case < len(test_cases):
            test_cases = [test_cases[args.test_case]]

    print(f"Using device: {device}")
    
    base_seed = int(args.seed) if args.seed is not None else int(time.time() * 1000) % (2**31)
    setup_seed(base_seed)
    print(f"[INFO] Evaluation seed base: {base_seed}")

    oscillation_file = None
    oscillation_writer = None
    if args.oscillation_csv:
        csv_dir = os.path.dirname(os.path.abspath(args.oscillation_csv))
        os.makedirs(csv_dir, exist_ok=True)
        oscillation_file = open(args.oscillation_csv, 'w', newline='', encoding='utf-8')
        oscillation_writer = csv.DictWriter(
            oscillation_file,
            fieldnames=OSCILLATION_FIELDS,
        )
        oscillation_writer.writeheader()
        oscillation_file.flush()
        print(f"[INFO] Oscillation metrics: {os.path.abspath(args.oscillation_csv)}")

    for case_idx, case in enumerate(test_cases):
        print(f"\n{'='*70}")
        print(f"Test Case [{case_idx}]: {case['desc']} | {case['human_num']} humans")
        print(f"{'='*70}")

        env_config = configparser.RawConfigParser()
        env_config.read(args.env_config)
        policy_config = configparser.RawConfigParser()
        policy_config.read(args.policy_config)
        
        # [FIX 1] Config Injection - MATCH TRAINING CONFIG
        if not policy_config.has_section('buffer'): policy_config.add_section('buffer')
        policy_config.set('buffer', 'seq_len', '12')  # Match training: T=12 (from train.config)

        if not policy_config.has_section('robot'): policy_config.add_section('robot')
        policy_config.set('robot', 'v_pref', '1.0')   # Match training

        # [FIX 2] Env Injection - MATCH TRAINING CONFIG
        if not env_config.has_section('env'): env_config.add_section('env')
        env_config.set('env', 'time_step', '0.25')  # Match training: dt=0.25
        env_config.set('env', 'time_limit', '25')

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

        # Use MambaRL_PPO directly (not from policy_factory)
        try:
            policy = MambaRL_PPO(policy_config, device=device)
        except Exception as exc:
            print(f"[WARNING] PPO config initialization failed, using defaults: {exc}")
            policy = MambaRL_PPO(device=device)

        weights_path = os.path.join(args.model_dir, args.weights)

        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get(
            'policy_state',
            checkpoint.get(
                'model_state_dict',
                checkpoint.get('value_state', checkpoint.get('model', checkpoint)),
            ),
        )
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        if hasattr(policy, 'load_state_dict'): policy.load_state_dict(state_dict)
        elif hasattr(policy, 'model'): policy.model.load_state_dict(state_dict)
        
        if hasattr(policy, 'to'): policy.to(device)
        policy.device = device
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
        successful_oscillation_rows = []
        time_step = env_config.getfloat('env', 'time_step', fallback=0.25)
        
        pbar = tqdm(range(args.episodes), ncols=100)
        for ep in pbar:
            episode_seed = (
                base_seed + case['case_id'] * 1_000_003 + ep
            ) % (2**31 - 1)
            
            outcome, steps, robot_positions, min_dists, oscillation_metrics = test_episode(
                env, robot, policy, episode_seed,
                robot_start=case.get('robot_start'),
                robot_goal=case.get('robot_goal')
            )

            if outcome == 'success':
                success_count += 1
                successful_oscillation_rows.append(oscillation_metrics)
            elif outcome == 'collision': collision_count += 1
            else: timeout_count += 1

            if oscillation_writer is not None:
                oscillation_writer.writerow({
                    'run_label': args.run_label,
                    'scenario': case['desc'],
                    'scenario_index': case['case_id'],
                    'episode': ep,
                    'seed': episode_seed,
                    'outcome': outcome,
                    'steps': steps,
                    'duration_s': steps * time_step,
                    **oscillation_metrics,
                })
                oscillation_file.flush()
            
            pbar.set_postfix({'S': f'{success_count}/{ep+1}', 'C': f'{collision_count}/{ep+1}'})

        print(f"\nResults for {case['desc']}:")
        print(f"  SUCCESS:   {success_count}/{args.episodes} ({100*success_count/args.episodes:.1f}%)")
        print(f"  COLLISION: {collision_count}/{args.episodes} ({100*collision_count/args.episodes:.1f}%)")
        print(f"  TIMEOUT:   {timeout_count}/{args.episodes} ({100*timeout_count/args.episodes:.1f}%)")
        if successful_oscillation_rows:
            print(
                "  OSC(success): "
                f"switch={np.mean([x['action_switch_hz'] for x in successful_oscillation_rows]):.3f}/s, "
                f"turn-rev={np.mean([x['turn_reversal_hz'] for x in successful_oscillation_rows]):.3f}/s, "
                f"curvature={np.mean([x['curvature_rad_per_m'] for x in successful_oscillation_rows]):.3f} rad/m, "
                f"lateral-sign={np.mean([x['lateral_sign_change_hz'] for x in successful_oscillation_rows]):.3f}/s"
            )
            print(
                "  OSC-HARM(success): "
                f"stall={np.mean([x['stalled_window_ratio'] for x in successful_oscillation_rows]):.3f}, "
                f"osc-stall={np.mean([x['oscillatory_stall_window_ratio'] for x in successful_oscillation_rows]):.3f}, "
                f"backtrack={np.mean([x['backtracking_ratio'] for x in successful_oscillation_rows]):.3f}, "
                f"progress-eff={np.mean([x['goal_progress_efficiency'] for x in successful_oscillation_rows]):.3f}"
            )

    if oscillation_file is not None:
        oscillation_file.close()
        print(f"[INFO] Oscillation CSV complete: {os.path.abspath(args.oscillation_csv)}")

if __name__ == '__main__':
    main()
