#!/usr/bin/env python3
"""Controlled CrowdNav evaluation for value and direct-action policies.

Runtime parameters are loaded from the selected environment and policy
configuration files. Optional observation perturbations affect policy inputs
only; simulator truth remains unchanged for rewards and outcome metrics.
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

from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState, ObservableState
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.contracts import (
    GRID,
    action_to_discrete_index,
    discrete_index_to_action,
    joint34_to_tokens,
)

OSCILLATION_FIELDS = [
    'run_label',
    'scenario',
    'scenario_index',
    'episode',
    'seed',
    'outcome',
    'steps',
    'duration_s',
    'min_clearance_m',
    'geometric_contact_steps',
    'geometric_contact_duration_s',
    'discomfort_steps',
    'mean_clearance_m',
    'mean_inference_ms',
    'p95_inference_ms',
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


def _make_policy_human_observations(sorted_humans, test_args, rng):
    """Build policy-only observations without modifying simulator truth."""
    states = [human.get_observable_state() for human in sorted_humans]
    noise_std = float(getattr(test_args, 'obs_noise_std', 0.0) or 0.0)
    if noise_std <= 0.0:
        return states

    noisy_states = []
    for state in states:
        dx, dy = rng.normal(0.0, noise_std, size=2)
        noisy_states.append(
            ObservableState(
                float(state.px + dx),
                float(state.py + dy),
                float(state.vx),
                float(state.vy),
                float(state.radius),
            )
        )
    return noisy_states

# ========== SARL Value Network for Discrete Action Search ==========
import torch.nn as nn

def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class SARLValueNetwork(nn.Module):
    """SARL's Value Network - exact copy for compatibility"""
    def __init__(self, input_dim=13, self_state_dim=6, mlp1_dims=[150, 100], mlp2_dims=[100, 50],
                 mlp3_dims=[150, 100, 100, 1], attention_dims=[100, 100, 1], with_global_state=True):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        mlp3_input_dim = mlp2_dims[-1] + self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)

    def forward(self, state):
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)
        if self.with_global_state:
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        features = mlp2_output.view(size[0], size[1], -1)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value

def rotate_state(self_state, human_state):
    """Rotate state to robot-centric coordinates (SARL standard)

    Input: self_state (FullState), human_state (ObservableState)
    Output: 13-dim rotated state matching SARL's rotate() output

    SARL rotate() expects 14-dim input:
    [px, py, vx, vy, radius, gx, gy, v_pref, theta, px1, py1, vx1, vy1, radius1]
      0   1   2   3    4      5   6     7      8     9   10   11   12     13

    SARL rotate() outputs 13-dim:
    [dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum]
    """
    # Goal direction
    dx = self_state.gx - self_state.px
    dy = self_state.gy - self_state.py
    dg = np.sqrt(dx**2 + dy**2)
    rot = np.arctan2(dy, dx)

    # Rotated self velocity
    vx = self_state.vx * np.cos(rot) + self_state.vy * np.sin(rot)
    vy = self_state.vy * np.cos(rot) - self_state.vx * np.sin(rot)

    # Rotated human position (relative to robot)
    px1 = (human_state.px - self_state.px) * np.cos(rot) + (human_state.py - self_state.py) * np.sin(rot)
    py1 = (human_state.py - self_state.py) * np.cos(rot) - (human_state.px - self_state.px) * np.sin(rot)

    # Rotated human velocity
    vx1 = human_state.vx * np.cos(rot) + human_state.vy * np.sin(rot)
    vy1 = human_state.vy * np.cos(rot) - human_state.vx * np.sin(rot)

    # Distance to human
    da = np.sqrt((self_state.px - human_state.px)**2 + (self_state.py - human_state.py)**2)

    # Radius sum
    radius_sum = self_state.radius + human_state.radius

    # theta is 0 for holonomic
    theta = 0.0

    return [dg, self_state.v_pref, theta, self_state.radius, vx, vy,
            px1, py1, vx1, vy1, human_state.radius, da, radius_sum]

def compute_scene_complexity(state):
    """计算场景复杂度指标

    Returns:
        n_humans: 人类数量
        min_ttc: 最小TTC（Time-To-Collision）
    """
    robot = state.self_state
    human_states = state.human_states

    n_humans = len(human_states)

    if n_humans == 0:
        return 0, float('inf')

    min_ttc = float('inf')
    for h in human_states:
        rel_x = h.px - robot.px
        rel_y = h.py - robot.py
        rel_vx = h.vx - robot.vx
        rel_vy = h.vy - robot.vy
        dist = np.sqrt(rel_x**2 + rel_y**2 + 1e-6)
        closing = -(rel_x * rel_vx + rel_y * rel_vy) / (dist + 1e-6)
        ttc = dist / (closing + 1e-6) if closing > 0.1 else dist * 10
        min_ttc = min(min_ttc, ttc)

    return n_humans, min_ttc

def sarl_discrete_search(state, sarl_value_net, device, time_step=0.25, gamma=0.9, v_pref=1.0):
    """Use SARL Value Network to evaluate discrete actions and return best action"""
    from crowd_sim.envs.utils.state import ObservableState, FullState

    # Build action space (SARL style: 5 speeds x 16 rotations = 80 actions)
    speeds = [(np.exp((i + 1) / 5) - 1) / (np.e - 1) * v_pref for i in range(5)]
    rotations = np.linspace(0, 2 * np.pi, 16, endpoint=False)

    action_space = [ActionXY(0, 0)]  # Include stop action
    for rotation in rotations:
        for speed in speeds:
            action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))

    self_state = state.self_state
    human_states = state.human_states

    max_value = float('-inf')
    best_action = action_space[0]

    for action in action_space:
        # 1. Propagate robot state
        next_px = self_state.px + action.vx * time_step
        next_py = self_state.py + action.vy * time_step
        next_self = FullState(next_px, next_py, action.vx, action.vy, self_state.radius,
                              self_state.gx, self_state.gy, self_state.v_pref, self_state.theta)

        # 2. Propagate human states (assume constant velocity)
        next_humans = []
        for h in human_states:
            nh_px = h.px + h.vx * time_step
            nh_py = h.py + h.vy * time_step
            next_humans.append(ObservableState(nh_px, nh_py, h.vx, h.vy, h.radius))

        # 3. Compute immediate reward
        dmin = float('inf')
        collision = False
        for h in next_humans:
            dist = np.sqrt((next_self.px - h.px)**2 + (next_self.py - h.py)**2) - next_self.radius - h.radius
            if dist < 0:
                collision = True
                break
            dmin = min(dmin, dist)

        reaching_goal = np.sqrt((next_self.px - next_self.gx)**2 + (next_self.py - next_self.gy)**2) < next_self.radius

        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1.0
        elif dmin < 0.2:
            reward = (dmin - 0.2) * 0.5 * time_step
        else:
            reward = 0.0

        # 4. Build rotated input for SARL network
        if len(next_humans) > 0:
            rotated_states = []
            for h in next_humans:
                rotated = rotate_state(next_self, h)
                rotated_states.append(rotated)

            # Pad to at least 1 human
            batch_input = torch.tensor(rotated_states, dtype=torch.float32).unsqueeze(0).to(device)

            # 5. Get value from SARL network
            with torch.no_grad():
                next_value = sarl_value_net(batch_input).item()
        else:
            next_value = 0.0

        # 6. Compute total value
        value = reward + (gamma ** (time_step * v_pref)) * next_value

        if value > max_value:
            max_value = value
            best_action = action

    return best_action

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


def test_episode(env, robot, policy, test_case_idx, robot_start=None, robot_goal=None, sarl_value_net=None, device=None, case_desc=None):
    if hasattr(policy, 'reset_episode_stats'):
        policy.reset_episode_stats()
    # reset progress history for test-time heuristics
    if hasattr(policy, "_dist_hist"):
        policy._dist_hist = []
    else:
        setattr(policy, "_dist_hist", [])

    # 统计混合策略使用情况
    mamba_count = 0
    sarl_count = 0

    case_size = int(getattr(env, 'case_size', {}).get('test', 1000))
    reset_result = env.reset(
        seed=test_case_idx,
        options={'test_case': int(test_case_idx) % max(1, case_size)},
    )
    if isinstance(reset_result, tuple):
        ob = reset_result[0]
    else:
        ob = reset_result

    behavior_scheduler = None
    behavior_policies = []
    test_args = getattr(policy, '_test_args', None)
    behavior_profile = str(
        getattr(test_args, 'behavior_profile', 'nominal')
    )
    if behavior_profile != 'nominal':
        from crowd_nav.bayesian_pilot.protocol import (
            BehaviorScheduler,
            InterventionORCA,
            PROFILES,
        )

        behavior_config = getattr(policy, '_behavior_env_config')
        for human in env.humans:
            intervention_policy = InterventionORCA(behavior_config)
            human.set_policy(intervention_policy)
            behavior_policies.append(intervention_policy)
        behavior_seed = (
            int(test_case_idx)
            + int(getattr(test_args, 'behavior_seed_offset', 104729))
        )
        behavior_scheduler = BehaviorScheduler(
            PROFILES[behavior_profile],
            seed=behavior_seed,
        )
        behavior_scheduler.reset(len(behavior_policies))

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
    obs_rng = np.random.default_rng(int(test_case_idx) + 7919)
    start_pos = tuple(float(x) for x in robot.get_position())
    goal_pos = (float(robot.gx), float(robot.gy))

    max_steps = 500

    while not done and steps < max_steps:
        if behavior_scheduler is not None:
            behavior_scheduler.advance(behavior_policies)

        # --- Manual Act with TTC-Sorting (与训练时contracts.py一致) ---
        # 1. Get full state
        robot_state = robot.get_full_state()

        # 2. Get sorted humans by TTC (与contracts.py第435行一致)
        humans = env.humans
        ttc_list = []
        for h in humans:
            rel_x = h.px - robot.px
            rel_y = h.py - robot.py
            rel_vx = h.vx - robot.vx
            rel_vy = h.vy - robot.vy
            dist = np.sqrt(rel_x**2 + rel_y**2 + 1e-6)
            closing = -(rel_x * rel_vx + rel_y * rel_vy) / (dist + 1e-6)
            # 与contracts.py完全一致：静止人用距离*10作为TTC
            ttc = dist / (closing + 1e-6) if closing > 0.1 else dist * 10
            ttc_list.append((ttc, h))
        min_ttc = min([x[0] for x in ttc_list]) if ttc_list else float('inf')
        # 按TTC升序排序（最危险的排前面）
        sorted_humans = [h for _, h in sorted(ttc_list, key=lambda x: x[0])]
        if hasattr(policy, 'set_fullcrowd_human_states'):
            policy.set_fullcrowd_human_states(
                [human.get_observable_state() for human in humans]
            )

        # 3. Construct JointState
        human_states = _make_policy_human_observations(
            sorted_humans,
            test_args,
            obs_rng,
        )
        state = JointState(robot_state, human_states)

        # 4. Predict (混合策略：Mamba-SARL Hybrid with Adaptive Gating)
        used_mamba = False
        measure_latency = bool(
            test_args and getattr(test_args, 'measure_latency', False)
        )
        synchronize_cuda = bool(
            measure_latency
            and getattr(test_args, 'gpu', False)
            and torch.cuda.is_available()
        )
        if synchronize_cuda:
            torch.cuda.synchronize()
        decision_start = time.perf_counter() if measure_latency else None

        if sarl_value_net is not None and policy is not None:
            # ========== 混合门控策略 ==========
            # 计算场景复杂度
            n_humans, min_ttc = compute_scene_complexity(state)

            # 门控阈值（可调参数）
            HUMAN_THRESHOLD = 7      # 人数超过7个认为是复杂场景
            TTC_THRESHOLD = 2.5      # TTC小于2.5s认为是危险场景

            # 决策：复杂/危险场景用SARL（慢但准），简单场景用Mamba（快）
            if n_humans > HUMAN_THRESHOLD or min_ttc < TTC_THRESHOLD:
                # 复杂场景：使用SARL Value网络做离散搜索
                action = sarl_discrete_search(state, sarl_value_net, device)
                sarl_count += 1
            else:
                # 简单场景：使用Mamba快速预测
                action = policy.predict(state)
                mamba_count += 1
                used_mamba = True
            # ==================================
        elif sarl_value_net is not None:
            # 纯SARL模式（无Mamba）
            action = sarl_discrete_search(state, sarl_value_net, device)
            sarl_count += 1
        elif test_args and test_args.gating and hasattr(policy, 'predict_with_gating'):
            # Mamba内置门控模式
            action = policy.predict_with_gating(
                state,
                human_threshold=test_args.human_threshold,
                ttc_threshold=test_args.ttc_threshold
            )
            used_mamba = True
        elif test_args and test_args.discrete_search and hasattr(policy, 'act_discrete'):
            # Mamba离散搜索模式
            action = policy.act_discrete(state)
            used_mamba = True
        else:
            # 默认连续动作（纯Mamba）
            action = policy.predict(state)
            used_mamba = True
        # --------------------------------------------------------------
        if measure_latency:
            if synchronize_cuda:
                torch.cuda.synchronize()
            policy._inference_times_ms.append(
                1000.0 * (time.perf_counter() - decision_start)
            )

        # Optional: goal-bias blend for hard scenarios (test-time heuristic)
        if used_mamba and test_args and test_args.mamba_bias and hasattr(action, 'vx'):
            alpha = test_args.bias_alpha
            stall_alpha = test_args.bias_stall_alpha
            min_clear_req = test_args.bias_min_clear
            min_ttc_req = test_args.bias_min_ttc
            speed_floor = test_args.bias_speed_floor

            if case_desc in test_args.bias_scenarios:
                # compute min clearance
                min_clear = float('inf')
                for h in humans:
                    d = np.linalg.norm(np.array([h.px - robot.px, h.py - robot.py])) - robot.radius - h.radius
                    if d < min_clear:
                        min_clear = d

                # track progress (for stall detection)
                dist_goal = np.linalg.norm(np.array([robot.gx - robot.px, robot.gy - robot.py]))
                policy._dist_hist.append(dist_goal)
                if len(policy._dist_hist) > test_args.bias_patience:
                    policy._dist_hist = policy._dist_hist[-test_args.bias_patience:]
                progress = policy._dist_hist[0] - policy._dist_hist[-1] if len(policy._dist_hist) >= 2 else 0.0
                late_stage = steps >= int(max_steps * test_args.bias_late_ratio)
                stalled = progress < test_args.bias_progress_eps and late_stage

                # late-stage relaxation (optional): allow a bit more risk to avoid timeout
                late_frac = 0.0
                if late_stage:
                    late_frac = (steps / max_steps - test_args.bias_late_ratio) / max(1e-6, (1.0 - test_args.bias_late_ratio))
                    late_frac = float(np.clip(late_frac, 0.0, 1.0))
                min_clear_eff = min_clear_req * (1.0 - test_args.bias_late_relax * late_frac)
                min_ttc_eff = min_ttc_req * (1.0 - test_args.bias_late_relax * late_frac)
                speed_floor_eff = speed_floor * (1.0 + test_args.bias_late_speed_boost * late_frac)
                speed_floor_eff = float(min(1.0, speed_floor_eff))

                if min_clear > min_clear_eff and min_ttc > min_ttc_eff:
                    dx = robot.gx - robot.px
                    dy = robot.gy - robot.py
                    norm = np.linalg.norm([dx, dy]) + 1e-6
                    goal_vx = robot.v_pref * dx / norm
                    goal_vy = robot.v_pref * dy / norm
                    alpha_use = stall_alpha if stalled else alpha

                    # deadline boost (late-stage ramp, only if very safe)
                    if test_args.bias_deadline_alpha > 0:
                        if late_stage and min_clear > test_args.bias_deadline_min_clear and min_ttc > test_args.bias_deadline_min_ttc:
                            alpha_use = max(alpha_use, test_args.bias_deadline_alpha * late_frac)

                    vx = (1 - alpha_use) * action.vx + alpha_use * goal_vx
                    vy = (1 - alpha_use) * action.vy + alpha_use * goal_vy

                    # speed floor to reduce dithering
                    speed = np.linalg.norm([vx, vy]) + 1e-6
                    if speed < speed_floor_eff * robot.v_pref:
                        vx = vx / speed * robot.v_pref * speed_floor_eff
                        vy = vy / speed * robot.v_pref * speed_floor_eff
                    elif speed > robot.v_pref:
                        vx = vx / speed * robot.v_pref
                        vy = vy / speed * robot.v_pref
                    # optional: avoid stop action if safe (late-stage or stalled)
                    if test_args.bias_no_stop:
                        stop_eps = test_args.bias_stop_eps * robot.v_pref
                        if np.linalg.norm([vx, vy]) < stop_eps:
                            vx = goal_vx
                            vy = goal_vy
                    action = ActionXY(vx, vy)

        # Optional: Mamba timeout rescue (safe, late-stage only)
        if test_args and test_args.mamba_rescue:
            # progress history
            if not hasattr(policy, "_dist_hist"):
                policy._dist_hist = []
            dist_goal = np.linalg.norm(np.array([robot.gx - robot.px, robot.gy - robot.py]))
            policy._dist_hist.append(dist_goal)
            if len(policy._dist_hist) > test_args.rescue_patience:
                policy._dist_hist = policy._dist_hist[-test_args.rescue_patience:]
            # check progress
            progress = policy._dist_hist[0] - policy._dist_hist[-1] if len(policy._dist_hist) >= 2 else 0.0
            late_stage = steps >= int(max_steps * test_args.rescue_late_ratio)
            if late_stage and progress < test_args.rescue_progress_eps and min_ttc > test_args.rescue_min_ttc:
                # try goal-directed action if predicted clearance is safe
                dx = robot.gx - robot.px
                dy = robot.gy - robot.py
                norm = np.linalg.norm([dx, dy]) + 1e-6
                speed = robot.v_pref * test_args.rescue_speed_scale
                cand = ActionXY(speed * dx / norm, speed * dy / norm)
                # predict clearance after 1 step (humans constant velocity)
                min_pred_clear = float('inf')
                next_rx = robot.px + cand.vx * env.time_step
                next_ry = robot.py + cand.vy * env.time_step
                for h in humans:
                    nx = h.px + h.vx * env.time_step
                    ny = h.py + h.vy * env.time_step
                    d = np.linalg.norm(np.array([nx - next_rx, ny - next_ry])) - robot.radius - h.radius
                    if d < min_pred_clear:
                        min_pred_clear = d
                if min_pred_clear > test_args.rescue_min_clear:
                    action = cand

        executed_actions.append((float(action.vx), float(action.vy)))
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
    policy._last_behavior_event_count = (
        0
        if behavior_scheduler is None
        else int(
            sum(
                count
                for name, count in behavior_scheduler.counts.items()
                if name != 'nominal'
            )
        )
    )
    return (
        outcome,
        steps,
        robot_positions,
        min_dists,
        mamba_count,
        sarl_count,
        oscillation_metrics,
    )

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
    parser.add_argument('--model_dir', type=str, default='runs/mamba_vl', help='Model directory')
    parser.add_argument('--weights', type=str, default='rl_model_ep10000_2.pth', help='Checkpoint name or "latest"')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--episodes', type=int, default=500, help='Episodes per scenario')
    parser.add_argument(
        '--time-limit',
        type=int,
        default=25,
        help='Episode time limit in seconds (default: 25)',
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=None,
        help='Override the deterministic test-case pool size',
    )
    parser.add_argument(
        '--case-block-by-seed',
        action='store_true',
        help='Use seed-indexed case blocks without a scenario offset',
    )
    parser.add_argument('--test_case', type=int, default=None, help='Run only specified test case')
    parser.add_argument(
        '--test-cases',
        type=str,
        default=None,
        help='Comma-separated scenario IDs, for example 0,3',
    )
    parser.add_argument(
        '--behavior-profile',
        choices=['nominal', 'train_nonstationary', 'heldout_nonstationary'],
        default='nominal',
        help='Deterministic pedestrian intervention protocol',
    )
    parser.add_argument(
        '--behavior-seed-offset',
        type=int,
        default=104729,
        help='Independent seed offset for pedestrian interventions',
    )
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument(
        '--obs-noise-std',
        type=float,
        default=0.0,
        help='Gaussian position-noise std for policy observations only',
    )
    parser.add_argument(
        '--measure-latency',
        action='store_true',
        help='Measure synchronized policy decision latency',
    )
    parser.add_argument(
        '--no_progress',
        action='store_true',
        help='Disable tqdm progress bars for batch evaluation',
    )
    parser.add_argument('--oscillation_csv', type=str, default=None,
                        help='Write one row per episode with oscillation metrics')
    parser.add_argument('--run_label', type=str, default=None,
                        help='Label stored in --oscillation_csv (e.g. full or discrete)')
    parser.add_argument('--seq_len', type=int, default=None, help='Override seq_len/T (e.g. 12 for GRU, 24 for Mamba T24)')
    parser.add_argument('--temporal-backbone', choices=['mamba', 'gru', 'bayes', 'bayes_mean'], default=None,
                        help='Override the temporal block; bayes_mean is the posterior-mean control')
    parser.add_argument('--distributional-risk-weight', type=float, default=None,
                        help='Override Bayesian distributional CVaR mixture weight')
    parser.add_argument('--distributional-cvar-alpha', type=float, default=None,
                        help='Override Bayesian distributional lower-tail fraction')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--direct_discrete', action='store_true',
                        help='Use direct q_head action selection instead of SARL-style value lookahead')
    # 方案1：离散动作搜索选项
    parser.add_argument('--discrete_search', action='store_true', help='Enable discrete action search (SARL-style)')
    parser.add_argument('--gating', action='store_true', help='Enable density gating (auto switch between continuous/discrete)')
    parser.add_argument('--human_threshold', type=int, default=8, help='Human count threshold for gating')
    parser.add_argument('--ttc_threshold', type=float, default=2.0, help='TTC threshold for gating')
    # 方案2：使用SARL Value网络做离散搜索
    parser.add_argument('--sarl_value', type=str, default=None, help='Path to SARL rl_model.pth for value-based discrete search')
    # Test-time bias (goal-directed blend for hard scenarios)
    parser.add_argument('--mamba_bias', action='store_true', help='Enable goal-bias blend for hard scenarios')
    parser.add_argument('--bias_alpha', type=float, default=0.25, help='Blend weight toward goal')
    parser.add_argument('--bias_min_clear', type=float, default=0.20, help='Min clearance to apply bias')
    parser.add_argument('--bias_min_ttc', type=float, default=1.5, help='Min TTC to apply bias')
    parser.add_argument('--bias_speed_floor', type=float, default=0.6, help='Min speed ratio when bias active')
    parser.add_argument('--bias_patience', type=int, default=8, help='Steps to measure progress')
    parser.add_argument('--bias_progress_eps', type=float, default=0.05, help='Min progress within window')
    parser.add_argument('--bias_late_ratio', type=float, default=0.5, help='Start stall check after ratio of max steps')
    parser.add_argument('--bias_stall_alpha', type=float, default=0.5, help='Stronger blend when stalled')
    parser.add_argument('--bias_deadline_alpha', type=float, default=0.0, help='Extra late-stage alpha ramp')
    parser.add_argument('--bias_deadline_min_clear', type=float, default=0.30, help='Min clearance for deadline boost')
    parser.add_argument('--bias_deadline_min_ttc', type=float, default=2.0, help='Min TTC for deadline boost')
    parser.add_argument('--bias_late_relax', type=float, default=0.0, help='Late-stage relaxation factor for clearance/TTC')
    parser.add_argument('--bias_late_speed_boost', type=float, default=0.0, help='Late-stage boost to speed floor (ratio)')
    parser.add_argument('--bias_no_stop', action='store_true', help='Avoid stop action when safe (goal-directed override)')
    parser.add_argument('--bias_stop_eps', type=float, default=0.05, help='Speed ratio treated as stop')
    parser.add_argument('--bias_scenarios', type=str, default='dense_square,large_circle,large_square',
                        help='Comma-separated scenario names to apply bias')
    # Mamba timeout rescue (late-stage, safe override)
    parser.add_argument('--mamba_rescue', action='store_true', help='Enable late-stage timeout rescue')
    parser.add_argument('--rescue_patience', type=int, default=8, help='Steps to measure progress')
    parser.add_argument('--rescue_progress_eps', type=float, default=0.05, help='Min progress within window')
    parser.add_argument('--rescue_min_clear', type=float, default=0.25, help='Min clearance to allow rescue')
    parser.add_argument('--rescue_min_ttc', type=float, default=2.0, help='Min TTC to allow rescue')
    parser.add_argument('--rescue_late_ratio', type=float, default=0.6, help='Start rescue after ratio of max steps')
    parser.add_argument('--rescue_speed_scale', type=float, default=1.0, help='Speed scale for rescue action')
    args = parser.parse_args()
    # normalize scenario list
    args.bias_scenarios = [s.strip() for s in args.bias_scenarios.split(',') if s.strip()]

    logging.basicConfig(level=logging.INFO)
    print("[INFO] ========================================")
    print("[INFO] TEST.PY - Mamba Policy Testing")
    print("[INFO] ========================================")
    print("[INFO] Config: T=24, dt=0.25, v_max=1.0 (MATCHED TO TRAINING)")
    print("[INFO] Feature: TTC-Sorting enabled (与contracts.py训练时一致)")
    if args.obs_noise_std > 0.0:
        print(
            f"[INFO] Policy-only position noise: "
            f"std={args.obs_noise_std:.3f} m"
        )
    if args.discrete_search:
        print("[INFO] Feature: Discrete action search ENABLED (SARL-style)")
    if args.gating:
        print(f"[INFO] Feature: Density gating ENABLED (human>{args.human_threshold} or ttc<{args.ttc_threshold}s -> discrete)")

    # Load SARL Value Network if specified
    sarl_value_net = None
    if args.sarl_value:
        print(f"[INFO] Loading SARL Value Network from {args.sarl_value}")
        sarl_value_net = SARLValueNetwork()
        sarl_weights = torch.load(args.sarl_value, map_location='cpu')
        sarl_value_net.load_state_dict(sarl_weights)
        sarl_value_net.eval()
        print("[INFO] SARL Value Network loaded successfully")
        print("[INFO] ========================================")
        print("[INFO] HYBRID MODE: Mamba-SARL Adaptive Gating")
        print("[INFO] - Simple scenes (≤7 humans, TTC>2.5s): Mamba (fast)")
        print("[INFO] - Complex scenes (>7 humans or TTC<2.5s): SARL (accurate)")
        print("[INFO] ========================================")

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    if sarl_value_net:
        sarl_value_net = sarl_value_net.to(device)

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
        else:
            raise ValueError(
                f"test_case must be in [0, {len(test_cases) - 1}]"
            )
    if args.test_cases is not None:
        if args.test_case is not None:
            raise ValueError(
                '--test_case and --test-cases are mutually exclusive'
            )
        selected_case_ids = {
            int(value.strip())
            for value in args.test_cases.split(',')
            if value.strip()
        }
        valid_case_ids = {case['case_id'] for case in test_cases}
        invalid_case_ids = selected_case_ids - valid_case_ids
        if invalid_case_ids:
            raise ValueError(
                f"Unknown test case IDs: {sorted(invalid_case_ids)}"
            )
        test_cases = [
            case
            for case in test_cases
            if case['case_id'] in selected_case_ids
        ]

    print(f"Using device: {device}")

    # Fixed seed makes two policy runs evaluate exactly the same cases.
    base_seed = int(args.seed) if args.seed is not None else int(time.time() * 1000) % (2**31)
    setup_seed(base_seed)
    print(f"[INFO] Evaluation seed base: {base_seed}")
    if args.case_block_by_seed:
        if args.test_size is None:
            raise ValueError(
                '--case-block-by-seed requires an explicit --test-size'
            )
        block_start = base_seed * args.episodes
        block_end = block_start + args.episodes
        if block_end > args.test_size:
            raise ValueError(
                f'Case block [{block_start}, {block_end}) exceeds '
                f'test-size {args.test_size}'
            )
        print(
            f"[INFO] Deterministic case block: "
            f"[{block_start}, {block_end}) / {args.test_size}"
        )

    oscillation_file = None
    oscillation_writer = None
    run_label = args.run_label or os.path.basename(os.path.normpath(args.model_dir))
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
        _seq_len = str(args.seq_len) if args.seq_len is not None else policy_config.get('temporal', 'T', fallback='12')
        policy_config.set('buffer', 'seq_len', _seq_len)
        if not policy_config.has_section('temporal'): policy_config.add_section('temporal')
        policy_config.set('temporal', 'T', _seq_len)

        if not policy_config.has_section('robot'): policy_config.add_section('robot')
        policy_config.set('robot', 'v_pref', '1.0')   # Match training

        # [FIX 2] Env Injection - MATCH TRAINING CONFIG
        if not env_config.has_section('env'): env_config.add_section('env')
        env_config.set('env', 'time_step', '0.25')  # Match training: dt=0.25
        env_config.set('env', 'time_limit', str(args.time_limit))
        if args.test_size is not None:
            if not 1 <= args.test_size <= 1000:
                raise ValueError('--test-size must be in [1, 1000]')
            env_config.set('env', 'test_size', str(args.test_size))

        # [FIX FINAL] MERGE EVERYTHING from env_config into policy_config
        # This mimics train.py behavior and solves all missing section errors.
        for section in env_config.sections():
            if not policy_config.has_section(section):
                policy_config.add_section(section)
            for key, value in env_config.items(section):
                # Only set if not already present (preserve policy specific overrides)
                if not policy_config.has_option(section, key):
                    policy_config.set(section, key, value)

        # [FIX SUPPLEMENT] Ensure manual injections are still present if not in env.config
        if not policy_config.has_section('train'): policy_config.add_section('train')
        if not policy_config.has_option('train', 'gamma'): policy_config.set('train', 'gamma', '0.99')

        if not policy_config.has_section('env'): policy_config.add_section('env')
        if not policy_config.has_option('env', 'time_step'): policy_config.set('env', 'time_step', '0.25')

        # [FIX 6] Minimal config injection for Mamba
        # Mamba only needs epsilon_start from [sarl] section (for compatibility)
        if not policy_config.has_section('sarl'):
            policy_config.add_section('sarl')
        policy_config.set('sarl', 'epsilon_start', '0.0')

        if args.temporal_backbone is not None:
            if not policy_config.has_section('mamba'):
                policy_config.add_section('mamba')
            policy_config.set('mamba', 'temporal_backbone', args.temporal_backbone)

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

        # ========================================
        # MAMBA POLICY LOADING (test.py specific)
        # ========================================
        try:
            print(f"[INFO] Loading Mamba policy: {args.policy}")
            policy_class = policy_factory[args.policy]
            policy = policy_class(policy_config)
        except Exception as e:
            print(f"[WARNING] Failed to init with config, trying without: {e}")
            policy = policy_class()
            if hasattr(policy, 'configure'):
                policy.configure(policy_config)

        # [FIX CRITICAL] Ensure device and phase are set
        if hasattr(policy, 'set_device'):
            policy.set_device(device)
        if hasattr(policy, 'set_phase'):
            policy.set_phase('test')
        if args.distributional_risk_weight is not None:
            if not 0.0 <= args.distributional_risk_weight <= 1.0:
                raise ValueError('--distributional-risk-weight must be in [0, 1]')
            if not hasattr(policy, 'risk_weight'):
                raise ValueError(
                    '--distributional-risk-weight requires a distributional policy'
                )
            policy.risk_weight = args.distributional_risk_weight
            print(
                '[INFO] Distributional risk weight override: '
                f'{policy.risk_weight:.3f}'
            )
        if args.distributional_cvar_alpha is not None:
            if not 0.0 < args.distributional_cvar_alpha <= 1.0:
                raise ValueError('--distributional-cvar-alpha must be in (0, 1]')
            if not hasattr(policy, 'cvar_alpha'):
                raise ValueError(
                    '--distributional-cvar-alpha requires a distributional policy'
                )
            policy.cvar_alpha = args.distributional_cvar_alpha
            print(
                '[INFO] Distributional CVaR alpha override: '
                f'{policy.cvar_alpha:.3f}'
            )

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
        state_dict = checkpoint.get('policy_state', checkpoint.get('model_state_dict', checkpoint.get('value_state', checkpoint.get('model', checkpoint))))
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # Auto-detect architecture from checkpoint and rebuild policy if needed
        ck_is_gru = any('weight_ih_l' in k for k in state_dict.keys())
        model_is_gru = any('weight_ih_l' in k for k, _ in policy.named_parameters())
        ck_is_bayes = 'temporal_encoder.retention_logits' in state_dict
        model_is_bayes = hasattr(policy.temporal_encoder, 'posterior')
        if ck_is_bayes:
            from crowd_nav.policy.bayes_temporal import BayesianTemporalEncoder
            weight = state_dict['temporal_encoder.observation.weight']
            configured_backbone = policy_config.get('mamba', 'temporal_backbone', fallback='mamba')
            mean_only = configured_backbone == 'bayes_mean'
            policy.temporal_encoder = BayesianTemporalEncoder(
                d_model=weight.shape[1], latent_dim=weight.shape[0], mean_only=mean_only
            ).to(device)
            policy.temporal_backbone = policy.temporal_encoder.backbone_name
        elif ck_is_gru != model_is_gru or model_is_bayes:
            from crowd_nav.policy.mamba_rl import GRUTemporalEncoder, MambaTemporalEncoder
            if ck_is_gru:
                policy.temporal_encoder = GRUTemporalEncoder(
                    d_model=policy.temporal_encoder.d_model if hasattr(policy.temporal_encoder, 'd_model') else 256,
                    n_layers=4, dropout=0.0
                ).to(device)
                print("[INFO] Auto-switched temporal encoder: Mamba -> GRU")
            else:
                d_model = next(iter(state_dict.values())).shape[-1] if state_dict else 256
                policy.temporal_encoder = MambaTemporalEncoder(
                    d_model=256, n_layers=4, d_state=64, d_conv=4, expand=2
                ).to(device)
                print("[INFO] Auto-switched temporal encoder: GRU -> Mamba")

        if hasattr(policy, 'load_state_dict'): policy.load_state_dict(state_dict)
        elif hasattr(policy, 'model'): policy.model.load_state_dict(state_dict)

        checkpoint_algo = str(checkpoint.get('algo', '')).lower() if isinstance(checkpoint, dict) else ''
        direct_discrete = bool(args.direct_discrete or checkpoint_algo == 'discrete_mamba')
        if hasattr(policy, 'use_sarl_predict'):
            policy.use_sarl_predict = not direct_discrete
            if direct_discrete:
                print("[INFO] Direct discrete prediction enabled (trained q_head)")
            else:
                print("[INFO] SARL-style value lookahead enabled")

        if hasattr(policy, 'to'): policy.to(device)
        if hasattr(policy, 'eval'): policy.eval()
        elif hasattr(policy, 'model'): policy.model.eval()

        print(f"[DEBUG] Policy Class: {type(policy)}")

        # 方案1：尝试加载Q网络（用于离散动作搜索）
        q_networks_loaded = False
        if args.discrete_search or args.gating:
            if hasattr(policy, 'load_q_networks'):
                # 优先尝试 value_pretrain.pth（Q网络专用checkpoint）
                q_ckpt_candidates = [
                    os.path.join(args.model_dir, 'value_pretrain.pth'),
                    weights_path,  # 回退到policy checkpoint
                ]
                for q_path in q_ckpt_candidates:
                    if os.path.exists(q_path):
                        q_networks_loaded = policy.load_q_networks(q_path)
                        if q_networks_loaded:
                            print(f"[INFO] Q networks loaded from {q_path}")
                            break
                if not q_networks_loaded:
                    print(f"[WARNING] Q networks not found, using continuous only")

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

        # 保存测试配置到policy（用于门控决策）
        if env_config.has_option('orca', 'safety_space'):
            env_config.set(
                'orca',
                'safety_space',
                env_config.get('orca', 'safety_space').split('#', 1)[0].strip(),
            )
        policy._test_args = args
        policy._behavior_env_config = env_config
        policy._inference_times_ms = []
        policy._q_networks_loaded = q_networks_loaded

        success_count = 0
        collision_count = 0
        timeout_count = 0
        total_mamba_steps = 0
        total_sarl_steps = 0
        success_times = []
        min_dist_sum = 0.0
        min_dist_steps = 0
        discomfort_events = 0
        successful_oscillation_rows = []
        behavior_event_total = 0
        time_step = env_config.getfloat('env', 'time_step', fallback=0.25)
        discomfort_dist = env_config.getfloat('reward', 'discomfort_dist', fallback=0.2)

        pbar = tqdm(
            range(args.episodes),
            desc=case['desc'],
            dynamic_ncols=True,
            mininterval=0.5,
            leave=False,
            disable=args.no_progress,
        )
        for ep in pbar:
            if args.case_block_by_seed:
                episode_seed = (
                    base_seed * args.episodes + ep
                ) % (2**31 - 1)
            else:
                episode_seed = (
                    base_seed * args.episodes
                    + case['case_id'] * 1_000_003
                    + ep
                ) % (2**31 - 1)

            latency_offset = len(policy._inference_times_ms)
            (
                outcome,
                steps,
                robot_positions,
                min_dists,
                mamba_count,
                sarl_count,
                oscillation_metrics,
            ) = test_episode(
                env, robot, policy, episode_seed,
                robot_start=case.get('robot_start'),
                robot_goal=case.get('robot_goal'),
                sarl_value_net=sarl_value_net,
                device=device,
                case_desc=case['desc'],
            )
            behavior_event_total += int(
                getattr(policy, '_last_behavior_event_count', 0)
            )

            if outcome == 'success':
                success_count += 1
                success_times.append(steps * time_step)
                successful_oscillation_rows.append(oscillation_metrics)
            elif outcome == 'collision': collision_count += 1
            else: timeout_count += 1

            if oscillation_writer is not None:
                episode_latencies = policy._inference_times_ms[latency_offset:]
                row = {
                    'run_label': run_label,
                    'scenario': case['desc'],
                    'scenario_index': case['case_id'],
                    'episode': ep,
                    'seed': episode_seed,
                    'outcome': outcome,
                    'steps': steps,
                    'duration_s': steps * time_step,
                    'min_clearance_m': float(np.min(min_dists)) if min_dists else float('inf'),
                    'geometric_contact_steps': sum(d < 0 for d in min_dists),
                    'geometric_contact_duration_s': sum(d < 0 for d in min_dists) * time_step,
                    'discomfort_steps': sum(d < discomfort_dist for d in min_dists),
                    'mean_clearance_m': float(np.mean(min_dists)) if min_dists else float('inf'),
                    'mean_inference_ms': float(np.mean(episode_latencies)) if episode_latencies else None,
                    'p95_inference_ms': float(np.percentile(episode_latencies, 95)) if episode_latencies else None,
                    **oscillation_metrics,
                }
                oscillation_writer.writerow(row)
                oscillation_file.flush()

            total_mamba_steps += mamba_count
            total_sarl_steps += sarl_count

            if min_dists:
                min_dist_sum += float(np.sum(min_dists))
                min_dist_steps += len(min_dists)
                discomfort_events += sum(1 for d in min_dists if d < discomfort_dist)

            pbar.set_postfix(
                {
                    'S': f'{success_count}/{ep + 1}',
                    'C': f'{collision_count}/{ep + 1}',
                },
                refresh=False,
            )

        print(f"\nResults for {case['desc']}:")
        print(f"  SUCCESS:   {success_count}/{args.episodes} ({100*success_count/args.episodes:.1f}%)")
        print(f"  COLLISION: {collision_count}/{args.episodes} ({100*collision_count/args.episodes:.1f}%)")
        print(f"  TIMEOUT:   {timeout_count}/{args.episodes} ({100*timeout_count/args.episodes:.1f}%)")
        avg_time = float(np.mean(success_times)) if success_times else 0.0
        avg_disc_freq = discomfort_events / args.episodes
        avg_disc_dist = (min_dist_sum / min_dist_steps) if min_dist_steps else float('inf')
        print(f"  TIME TAKEN (s): {avg_time:.2f}" if success_times else "  TIME TAKEN (s): n/a")
        print(f"  DISC. FREQ: {avg_disc_freq:.2f}")
        print(f"  DISC. DIST (m): {avg_disc_dist:.2f}")
        if args.behavior_profile != 'nominal':
            print(
                f"  BEHAVIOR:   profile={args.behavior_profile} "
                f"events={behavior_event_total}"
            )
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

        # 打印混合策略统计
        if sarl_value_net is not None and policy is not None:
            total_steps = total_mamba_steps + total_sarl_steps
            if total_steps > 0:
                mamba_pct = 100 * total_mamba_steps / total_steps
                sarl_pct = 100 * total_sarl_steps / total_steps
                print(f"  HYBRID:    Mamba {mamba_pct:.1f}% | SARL {sarl_pct:.1f}% (of {total_steps} steps)")
        if hasattr(policy, '_bma_weight_count'):
            weight_count = int(policy._bma_weight_count)
            mean_weight = (
                float(policy._bma_weight_sum) / weight_count
                if weight_count
                else float('nan')
            )
            high_fraction = (
                float(policy._bma_high_weight_count) / weight_count
                if weight_count
                else float('nan')
            )
            mean_max_weight = (
                float(policy._bma_max_weight_sum) / weight_count
                if weight_count
                else float('nan')
            )
            any_high_fraction = (
                float(policy._bma_any_high_weight_count) / weight_count
                if weight_count
                else float('nan')
            )
            print(
                "  BMA:        "
                f"mean_GDBN_weight={mean_weight:.4f} "
                f"mean_max_GDBN_weight={mean_max_weight:.4f} "
                f"high_weight_fraction={high_fraction:.4f} "
                f"any_high_fraction={any_high_fraction:.4f} "
                f"steps={weight_count}"
            )

    if oscillation_file is not None:
        oscillation_file.close()
        print(f"[INFO] Oscillation CSV complete: {os.path.abspath(args.oscillation_csv)}")

    if args.measure_latency and hasattr(policy, '_inference_times_ms'):
        times = np.asarray(policy._inference_times_ms, dtype=np.float64)
        if len(times):
            print(
                "[LATENCY] "
                f"n={len(times)} mean={np.mean(times):.3f} ms "
                f"median={np.median(times):.3f} ms "
                f"p95={np.percentile(times, 95):.3f} ms "
                f"max={np.max(times):.3f} ms"
            )

if __name__ == '__main__':
    main()
