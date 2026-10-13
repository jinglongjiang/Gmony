#!/usr/bin/env python3
"""Evaluate one-step value-ranking consistency under drive conversion.

For each sampled CrowdNav state, this script scores every discrete action twice:

1. the nominal holonomic successor used by Mamba-VL; and
2. the unicycle successor produced by the Gazebo deployment controller.

The test reports top-action agreement, top-k overlap, and the value regret of
executing the nominally selected action under the differential-drive model.
It is an evaluation-only analysis; no training or checkpoint updates occur.
"""

import argparse
import configparser
import csv
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# Prefer the repository containing this script over stale PYTHONPATH entries.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import FullState, JointState, ObservableState


SCENARIOS = [
    {'case_id': 0, 'name': 'baseline_circle', 'sim': 'circle_crossing', 'human_num': 5,
     'circle_radius': 4.0},
    {'case_id': 1, 'name': 'baseline_square', 'sim': 'square_crossing', 'human_num': 10,
     'square_width': 10.0},
    {'case_id': 2, 'name': 'dense_circle', 'sim': 'circle_crossing', 'human_num': 10,
     'circle_radius': 4.0},
    {'case_id': 3, 'name': 'dense_square', 'sim': 'square_crossing', 'human_num': 20,
     'square_width': 10.0},
    {'case_id': 4, 'name': 'large_circle', 'sim': 'circle_crossing', 'human_num': 12,
     'circle_radius': 6.0},
    {'case_id': 5, 'name': 'large_square', 'sim': 'square_crossing', 'human_num': 20,
     'square_width': 14.0},
]


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def merge_missing(dst, src):
    for section in src.sections():
        if not dst.has_section(section):
            dst.add_section(section)
        for key, value in src.items(section):
            if not dst.has_option(section, key):
                dst.set(section, key, value)


def build_policy_config(args):
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)

    if not policy_config.has_section('buffer'):
        policy_config.add_section('buffer')
    policy_config.set('buffer', 'seq_len', str(args.seq_len))

    if not policy_config.has_section('temporal'):
        policy_config.add_section('temporal')
    policy_config.set('temporal', 'T', str(args.seq_len))

    if not policy_config.has_section('mamba'):
        policy_config.add_section('mamba')
    if args.temporal_backbone:
        policy_config.set('mamba', 'temporal_backbone', args.temporal_backbone)

    if not policy_config.has_section('robot'):
        policy_config.add_section('robot')
    policy_config.set('robot', 'v_pref', '1.0')

    if not env_config.has_section('env'):
        env_config.add_section('env')
    env_config.set('env', 'time_step', str(args.holonomic_dt))
    env_config.set('env', 'time_limit', str(int(args.time_limit)))

    merge_missing(policy_config, env_config)

    if not policy_config.has_section('train'):
        policy_config.add_section('train')
    if not policy_config.has_option('train', 'gamma'):
        policy_config.set('train', 'gamma', '0.99')

    if not policy_config.has_section('sarl'):
        policy_config.add_section('sarl')
    policy_config.set('sarl', 'epsilon_start', '0.0')
    return policy_config


def load_policy(args, device):
    config = build_policy_config(args)
    policy = policy_factory[args.policy](config)
    if hasattr(policy, 'set_device'):
        policy.set_device(device)
    else:
        policy.device = device
        policy.to(device)
    if hasattr(policy, 'set_phase'):
        policy.set_phase('test')
    else:
        policy._phase = 'test'
    if hasattr(policy, 'set_env_dt'):
        policy.set_env_dt(args.holonomic_dt)

    weights_path = os.path.join(args.model_dir, args.weights)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get(
        'policy_state',
        checkpoint.get(
            'model_state_dict',
            checkpoint.get('value_state', checkpoint.get('model', checkpoint)),
        ),
    )
    if any(key.startswith('_orig_mod.') for key in state_dict):
        state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}

    checkpoint_algo = str(checkpoint.get('algo', '')).lower()
    if checkpoint_algo == 'discrete_mamba':
        raise ValueError(
            'This analysis requires a value-lookahead checkpoint, but the checkpoint is direct-Q.'
        )

    policy.load_state_dict(state_dict)
    policy.use_sarl_predict = True
    policy.to(device)
    policy.eval()
    return policy, weights_path, checkpoint_algo


def build_env_config(args, scenario, policy_config):
    config = configparser.RawConfigParser()
    config.read(args.env_config)

    if not config.has_section('env'):
        config.add_section('env')
    config.set('env', 'time_step', str(args.holonomic_dt))
    config.set('env', 'time_limit', str(int(args.time_limit)))

    if not config.has_section('robot'):
        config.add_section('robot')
    config.set('robot', 'v_pref', '1.0')

    for section in policy_config.sections():
        if not config.has_section(section):
            config.add_section(section)
        for key, value in policy_config.items(section):
            config.set(section, key, value)

    if not config.has_section('sim'):
        config.add_section('sim')
    config.set('sim', 'test_sim', scenario['sim'])
    config.set('sim', 'human_num', str(scenario['human_num']))
    if 'circle_radius' in scenario:
        config.set('sim', 'circle_radius', str(scenario['circle_radius']))
    if 'square_width' in scenario:
        config.set('sim', 'square_width', str(scenario['square_width']))
    return config


def sorted_joint_state(robot, humans):
    ranked = []
    for human in humans:
        rel_x = human.px - robot.px
        rel_y = human.py - robot.py
        rel_vx = human.vx - robot.vx
        rel_vy = human.vy - robot.vy
        distance = math.sqrt(rel_x * rel_x + rel_y * rel_y + 1e-6)
        closing = -(rel_x * rel_vx + rel_y * rel_vy) / (distance + 1e-6)
        ttc = distance / (closing + 1e-6) if closing > 0.1 else distance * 10.0
        ranked.append((ttc, human.get_observable_state()))
    ranked.sort(key=lambda item: item[0])
    return JointState(robot.get_full_state(), [state for _, state in ranked])


def propagate_humans(humans, dt):
    return [
        ObservableState(
            human.px + human.vx * dt,
            human.py + human.vy * dt,
            human.vx,
            human.vy,
            human.radius,
        )
        for human in humans
    ]


def controller_command(action, robot, yaw, min_center_distance, args):
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    vx_body = action.vx * cos_yaw + action.vy * sin_yaw
    vy_body = -action.vx * sin_yaw + action.vy * cos_yaw
    forward = max(0.0, vx_body)
    action_heading = math.atan2(vy_body, max(0.2, forward))

    goal_heading = wrap_angle(math.atan2(robot.gy - robot.py, robot.gx - robot.px) - yaw)
    heading = args.action_heading_weight * action_heading
    heading += (1.0 - args.action_heading_weight) * goal_heading
    heading = float(np.clip(heading, -args.heading_limit, args.heading_limit))

    linear = args.cruise_speed * max(args.speed_cos_floor, math.cos(heading))
    if min_center_distance < args.near_human_distance:
        linear = min(linear, args.near_human_speed)
    else:
        linear = max(linear, args.far_speed_floor)
    linear = float(np.clip(linear, args.min_speed, args.cruise_speed))
    angular = float(np.clip(args.heading_gain * heading, -args.omega_max, args.omega_max))
    return linear, angular


def unicycle_successor(robot, yaw, linear, angular, dt):
    if abs(angular) < 1e-8:
        dx_body = linear * dt
        dy_body = 0.0
    else:
        dx_body = (linear / angular) * math.sin(angular * dt)
        dy_body = (linear / angular) * (1.0 - math.cos(angular * dt))

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    dx_world = cos_yaw * dx_body - sin_yaw * dy_body
    dy_world = sin_yaw * dx_body + cos_yaw * dy_body
    yaw_next = wrap_angle(yaw + angular * dt)
    vx_world = linear * math.cos(yaw_next)
    vy_world = linear * math.sin(yaw_next)
    return FullState(
        robot.px + dx_world,
        robot.py + dy_world,
        vx_world,
        vy_world,
        robot.radius,
        robot.gx,
        robot.gy,
        robot.v_pref,
        yaw_next,
    )


def clearance(robot, humans):
    if not humans:
        return float('inf')
    return min(
        math.hypot(robot.px - human.px, robot.py - human.py) - robot.radius - human.radius
        for human in humans
    )


def apply_eval_adjustments(policy, values, clearances):
    adjusted = values.clone()
    clearances_tensor = torch.as_tensor(clearances, dtype=adjusted.dtype, device=adjusted.device)
    if policy.test_min_clearance > 0.0:
        safe = clearances_tensor >= policy.test_min_clearance
        if bool(safe.any()):
            adjusted = adjusted.masked_fill(~safe, -1e9)
    if policy.test_risk_lambda > 0.0:
        margin = (
            policy.test_min_clearance
            if policy.test_min_clearance > 0.0
            else policy.discomfort_dist
        )
        adjusted = adjusted - policy.test_risk_lambda * torch.clamp(
            margin - clearances_tensor, min=0.0
        )
    if adjusted.numel() > 0:
        adjusted[0] -= 1e-3
    return adjusted


def make_sequence(policy, base_history, next_robot, next_humans):
    state_34 = policy._build_joint_state_34(next_robot, next_humans)
    token = _batch_joint34_to_tokens_vectorized(state_34.reshape(1, -1))[0]
    history = base_history + [token]
    history = history[-policy.seq_len:]
    if len(history) < policy.seq_len:
        history = [history[0]] * (policy.seq_len - len(history)) + history
    return np.asarray(history, dtype=np.float32)


def score_paired_successors(policy, state, yaw, args):
    if policy.action_space is None:
        policy.build_action_space(state.self_state.v_pref)

    current_34 = policy._build_joint_state_34(state.self_state, state.human_states)
    current_token = _batch_joint34_to_tokens_vectorized(current_34.reshape(1, -1))[0]
    base_history = list(policy._history) + [current_token]

    min_center_distance = min(
        (math.hypot(state.self_state.px - human.px, state.self_state.py - human.py)
         for human in state.human_states),
        default=float('inf'),
    )
    humans_holonomic = propagate_humans(state.human_states, args.holonomic_dt)
    humans_diffdrive = propagate_humans(state.human_states, args.plan_dt)

    sequences = []
    rewards_h = []
    rewards_d = []
    clearances_h = []
    clearances_d = []

    for action in policy.action_space:
        next_h = FullState(
            state.self_state.px + action.vx * args.holonomic_dt,
            state.self_state.py + action.vy * args.holonomic_dt,
            action.vx,
            action.vy,
            state.self_state.radius,
            state.self_state.gx,
            state.self_state.gy,
            state.self_state.v_pref,
            state.self_state.theta,
        )
        linear, angular = controller_command(
            action, state.self_state, yaw, min_center_distance, args
        )
        next_d = unicycle_successor(
            state.self_state, yaw, linear, angular, args.plan_dt
        )
        executed_action = ActionXY(next_d.vx, next_d.vy)

        rewards_h.append(
            policy.compute_reward(
                next_h,
                humans_holonomic,
                prev_nav=state.self_state,
                action=action,
            )
        )
        rewards_d.append(
            policy.compute_reward(
                next_d,
                humans_diffdrive,
                prev_nav=state.self_state,
                action=executed_action,
            )
        )
        clearances_h.append(clearance(next_h, humans_holonomic))
        clearances_d.append(clearance(next_d, humans_diffdrive))
        sequences.append(make_sequence(policy, base_history, next_h, humans_holonomic))
        sequences.append(make_sequence(policy, base_history, next_d, humans_diffdrive))

    tensor = torch.from_numpy(np.asarray(sequences)).float().to(policy.device)
    with torch.inference_mode():
        values = policy.forward_value(tensor)
    value_h = values[0::2]
    value_d = values[1::2]
    reward_h = torch.as_tensor(rewards_h, dtype=value_h.dtype, device=policy.device)
    reward_d = torch.as_tensor(rewards_d, dtype=value_d.dtype, device=policy.device)
    score_h = reward_h + policy.gamma * value_h
    score_d = reward_d + policy.gamma * value_d
    decision_h = apply_eval_adjustments(policy, score_h, clearances_h)
    decision_d = apply_eval_adjustments(policy, score_d, clearances_d)
    return {
        'current_token': current_token,
        'score_h': score_h,
        'score_d': score_d,
        'decision_h': decision_h,
        'decision_d': decision_d,
        'min_center_distance': min_center_distance,
    }


def ranking_metrics(scores):
    score_h = scores['score_h']
    score_d = scores['score_d']
    decision_h = scores['decision_h']
    decision_d = scores['decision_d']

    raw_h = int(torch.argmax(score_h).item())
    raw_d = int(torch.argmax(score_d).item())
    chosen_h = int(torch.argmax(decision_h).item())
    chosen_d = int(torch.argmax(decision_d).item())
    k = min(5, score_h.numel())
    top_h = set(torch.topk(score_h, k=k).indices.detach().cpu().tolist())
    top_d = set(torch.topk(score_d, k=k).indices.detach().cpu().tolist())

    sorted_h = torch.topk(score_h, k=min(2, score_h.numel())).values
    margin_h = float((sorted_h[0] - sorted_h[1]).item()) if sorted_h.numel() > 1 else 0.0
    max_delta = float(torch.max(torch.abs(score_h - score_d)).item())
    regret = float((torch.max(score_d) - score_d[raw_h]).item())
    score_span = float((torch.max(score_d) - torch.min(score_d)).item())
    normalized_regret = regret / max(score_span, 1e-8)

    return {
        'raw_argmax_h': raw_h,
        'raw_argmax_d': raw_d,
        'raw_argmax_agree': int(raw_h == raw_d),
        'decision_argmax_h': chosen_h,
        'decision_argmax_d': chosen_d,
        'decision_argmax_agree': int(chosen_h == chosen_d),
        'top5_overlap': len(top_h.intersection(top_d)) / float(k),
        'diffdrive_regret': max(0.0, regret),
        'normalized_regret': max(0.0, normalized_regret),
        'holonomic_margin': margin_h,
        'max_score_delta': max_delta,
        'margin_coverage': int(margin_h > 2.0 * max_delta),
    }


def restore_policy_state(policy, history, last_action):
    policy._history.clear()
    policy._history.extend(history)
    policy._last_action = last_action


def recover_unsmoothed_index(policy, executed_action, previous_action):
    alpha = float(policy.test_action_smoothing)
    if alpha > 0.0 and previous_action is not None and alpha < 1.0:
        vx = (executed_action.vx - alpha * previous_action.vx) / (1.0 - alpha)
        vy = (executed_action.vy - alpha * previous_action.vy) / (1.0 - alpha)
        action = ActionXY(vx, vy)
    else:
        action = executed_action
    return int(policy.continuous_to_discrete_index(action))


def initial_yaw(robot, mode):
    if mode == 'goal':
        return math.atan2(robot.gy - robot.py, robot.gx - robot.px)
    if mode == 'zero':
        return 0.0
    return float(robot.theta)


def update_shadow_yaw(yaw, action, robot, humans, args):
    min_center_distance = min(
        (math.hypot(robot.px - human.px, robot.py - human.py) for human in humans),
        default=float('inf'),
    )
    if humans and min_center_distance < args.activation_distance:
        _, angular = controller_command(action, robot, yaw, min_center_distance, args)
    else:
        goal_error = wrap_angle(math.atan2(robot.gy - robot.py, robot.gx - robot.px) - yaw)
        angular = float(np.clip(args.heading_gain * goal_error, -args.omega_max, args.omega_max))
    return wrap_angle(yaw + angular * args.plan_dt)


def classify_outcome(info):
    name = info.__class__.__name__.lower()
    if 'reachgoal' in name:
        return 'success'
    if 'collision' in name:
        return 'collision'
    if isinstance(info, dict):
        event = str(info.get('event', '')).lower()
        if 'success' in event or 'reach_goal' in event:
            return 'success'
        if 'collision' in event:
            return 'collision'
    return 'timeout'


def run_episode(policy, env, robot, scenario, episode, episode_seed, args):
    policy.reset_episode_stats()
    result = env.reset(seed=episode_seed)
    if isinstance(result, tuple):
        result = result[0]
    del result

    yaw = initial_yaw(robot, args.initial_yaw)
    done = False
    step = 0
    rows = []
    info = {}
    max_steps = int(math.ceil(args.time_limit / args.holonomic_dt)) + 1

    while not done and step < max_steps:
        state = sorted_joint_state(robot, env.humans)
        min_center_distance = min(
            (math.hypot(robot.px - human.px, robot.py - human.py) for human in env.humans),
            default=float('inf'),
        )
        sample = (
            step % args.sample_every == 0
            and min_center_distance < args.activation_distance
            and (args.max_samples_per_episode <= 0 or len(rows) < args.max_samples_per_episode)
        )

        if sample:
            history_before = list(policy._history)
            last_action_before = policy._last_action

            # Advance the episode with the policy's unmodified implementation.
            reference_action = policy.predict(state)
            history_after = list(policy._history)
            last_action_after = policy._last_action
            reference_index = recover_unsmoothed_index(
                policy, reference_action, last_action_before
            )

            # Score the paired successors from the exact pre-decision history.
            restore_policy_state(policy, history_before, last_action_before)
            scores = score_paired_successors(policy, state, yaw, args)
            metrics = ranking_metrics(scores)
            metrics['reference_argmax'] = reference_index
            metrics['nominal_parity'] = int(
                reference_index == metrics['decision_argmax_h']
            )
            restore_policy_state(policy, history_after, last_action_after)
            action = reference_action
            rows.append({
                'scenario': scenario['name'],
                'case_id': scenario['case_id'],
                'episode': episode,
                'seed': episode_seed,
                'step': step,
                'yaw_rad': yaw,
                'human_num': len(state.human_states),
                'min_center_distance_m': scores['min_center_distance'],
                **metrics,
            })
        else:
            action = policy.predict(state)

        yaw = update_shadow_yaw(yaw, action, state.self_state, state.human_states, args)
        step_result = env.step(action)
        if len(step_result) == 5:
            _, _, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
        else:
            _, _, done, info = step_result
        step += 1

    return rows, classify_outcome(info)


def summarize(rows, outcomes):
    groups = defaultdict(list)
    for row in rows:
        groups[row['scenario']].append(row)
        groups['aggregate'].append(row)

    summary = {}
    for name, items in groups.items():
        summary[name] = {
            'sample_count': len(items),
            'raw_argmax_agreement': float(np.mean([x['raw_argmax_agree'] for x in items])),
            'decision_argmax_agreement': float(
                np.mean([x['decision_argmax_agree'] for x in items])
            ),
            'nominal_parity': float(np.mean([x['nominal_parity'] for x in items])),
            'mean_top5_overlap': float(np.mean([x['top5_overlap'] for x in items])),
            'mean_diffdrive_regret': float(np.mean([x['diffdrive_regret'] for x in items])),
            'p95_diffdrive_regret': float(
                np.percentile([x['diffdrive_regret'] for x in items], 95)
            ),
            'mean_normalized_regret': float(
                np.mean([x['normalized_regret'] for x in items])
            ),
            'margin_coverage': float(np.mean([x['margin_coverage'] for x in items])),
        }

    outcome_summary = {}
    for scenario, values in outcomes.items():
        total = max(1, len(values))
        outcome_summary[scenario] = {
            key: values.count(key) / total for key in ('success', 'collision', 'timeout')
        }
    return summary, outcome_summary


def write_results(args, rows, summary, outcomes, metadata):
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    csv_path = os.path.splitext(output_path)[0] + '_states.csv'

    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(
            {
                'metadata': metadata,
                'summary': summary,
                'rollout_outcomes': outcomes,
            },
            handle,
            indent=2,
        )

    if rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return output_path, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', default='mamba')
    parser.add_argument('--model_dir', default='runs/mamba_vl')
    parser.add_argument('--weights', default='rl_model_ep10000_2.pth')
    parser.add_argument('--env_config', default='configs/env.config')
    parser.add_argument('--policy_config', default='configs/policy.config')
    parser.add_argument('--temporal-backbone', choices=['mamba', 'gru'], default='mamba')
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--episodes', type=int, default=20, help='Episodes per scenario')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample_every', type=int, default=5)
    parser.add_argument('--max_samples_per_episode', type=int, default=20)
    parser.add_argument('--time_limit', type=float, default=25.0)
    parser.add_argument('--holonomic_dt', type=float, default=0.25)
    parser.add_argument('--plan_dt', type=float, default=0.20)
    parser.add_argument('--initial_yaw', choices=['state', 'goal', 'zero'], default='state')
    parser.add_argument('--activation_distance', type=float, default=4.0)
    parser.add_argument('--cruise_speed', type=float, default=0.8)
    parser.add_argument('--min_speed', type=float, default=0.10)
    parser.add_argument('--near_human_distance', type=float, default=0.8)
    parser.add_argument('--near_human_speed', type=float, default=0.18)
    parser.add_argument('--far_speed_floor', type=float, default=0.45)
    parser.add_argument('--speed_cos_floor', type=float, default=0.45)
    parser.add_argument('--action_heading_weight', type=float, default=0.65)
    parser.add_argument('--heading_gain', type=float, default=3.8)
    parser.add_argument('--heading_limit', type=float, default=1.0)
    parser.add_argument('--omega_max', type=float, default=3.0)
    parser.add_argument(
        '--output',
        default='runs/mamba_vl/diffdrive_consistency.json',
    )
    args = parser.parse_args()

    if args.sample_every < 1:
        raise ValueError('--sample_every must be at least 1')
    if not 0.0 <= args.action_heading_weight <= 1.0:
        raise ValueError('--action_heading_weight must be in [0, 1]')

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    if args.gpu and device.type != 'cuda':
        raise RuntimeError('--gpu was requested, but CUDA is not available')
    seed_everything(args.seed)

    policy_config = build_policy_config(args)
    policy, weights_path, checkpoint_algo = load_policy(args, device)
    scenarios = SCENARIOS
    if args.test_case is not None:
        scenarios = [scenario for scenario in SCENARIOS if scenario['case_id'] == args.test_case]
        if not scenarios:
            raise ValueError(f'Unknown --test_case {args.test_case}')

    print('[INFO] Holonomic-vs-differential value-ranking consistency')
    print(f'[INFO] device={device}, checkpoint={weights_path}')
    print(
        f'[INFO] horizons: holonomic={args.holonomic_dt:.2f}s, '
        f'differential={args.plan_dt:.2f}s; sample_every={args.sample_every}'
    )

    rows = []
    outcomes = defaultdict(list)
    progress_stream = sys.stderr
    close_progress_stream = False
    if not sys.stderr.isatty():
        try:
            progress_stream = open('/dev/tty', 'w', buffering=1)
            close_progress_stream = True
        except OSError:
            progress_stream = sys.stderr
    progress_bar = tqdm(
        total=len(scenarios) * args.episodes,
        desc='all_scenarios',
        unit='ep',
        dynamic_ncols=True,
        file=progress_stream,
    )
    for scenario in scenarios:
        env_config = build_env_config(args, scenario, policy_config)
        env = CrowdSim()
        env.configure(env_config)
        env.phase = 'test'
        robot = Robot(env_config, 'robot')
        robot.set_policy(policy)
        robot.env = env
        env.set_robot(robot)
        if hasattr(policy, 'set_env'):
            policy.set_env(env)

        print(
            f"[INFO] scenario={scenario['name']} humans={scenario['human_num']} "
            f'episodes={args.episodes}',
            flush=True,
        )
        scenario_rows = []
        agreement_count = 0
        parity_count = 0
        top5_sum = 0.0
        regret_sum = 0.0
        progress_bar.set_description(scenario['name'])
        for episode in range(args.episodes):
            episode_seed = (
                args.seed + scenario['case_id'] * 1_000_003 + episode
            ) % (2**31 - 1)
            episode_rows, outcome = run_episode(
                policy, env, robot, scenario, episode, episode_seed, args
            )
            scenario_rows.extend(episode_rows)
            outcomes[scenario['name']].append(outcome)
            agreement_count += sum(row['decision_argmax_agree'] for row in episode_rows)
            parity_count += sum(row['nominal_parity'] for row in episode_rows)
            top5_sum += sum(row['top5_overlap'] for row in episode_rows)
            regret_sum += sum(row['diffdrive_regret'] for row in episode_rows)
            sample_count = len(scenario_rows)
            progress_bar.set_postfix(
                local=f'{episode + 1}/{args.episodes}',
                n=sample_count,
                P=f'{100.0 * parity_count / max(1, sample_count):.1f}%',
                A=f'{100.0 * agreement_count / max(1, sample_count):.1f}%',
                T5=f'{100.0 * top5_sum / max(1, sample_count):.1f}%',
                R=f'{regret_sum / max(1, sample_count):.4g}',
                O=outcome[0].upper(),
            )
            progress_bar.update(1)
        rows.extend(scenario_rows)
        if scenario_rows:
            agreement = np.mean([row['decision_argmax_agree'] for row in scenario_rows])
            overlap = np.mean([row['top5_overlap'] for row in scenario_rows])
            regret = np.mean([row['diffdrive_regret'] for row in scenario_rows])
            print(
                f'[RESULT] {scenario["name"]}: samples={len(scenario_rows)}, '
                f'argmax={100.0 * agreement:.1f}%, top5={100.0 * overlap:.1f}%, '
                f'regret={regret:.6f}'
            )
        else:
            print(f'[WARNING] {scenario["name"]}: no states met the sampling condition')

    progress_bar.close()
    if close_progress_stream:
        progress_stream.close()

    if not rows:
        raise RuntimeError('No states were sampled; increase --activation_distance or episode count')

    summary, outcome_summary = summarize(rows, outcomes)
    metadata = {
        'checkpoint': os.path.abspath(weights_path),
        'checkpoint_algo': checkpoint_algo,
        'device': str(device),
        'seed': args.seed,
        'episodes_per_scenario': args.episodes,
        'state_source': 'holonomic CrowdNav closed-loop rollouts',
        'yaw_source': f'shadow differential-drive controller initialized from {args.initial_yaw}',
        'action_count': len(policy.action_space),
        'holonomic_horizon_s': args.holonomic_dt,
        'differential_horizon_s': args.plan_dt,
        'controller': {
            'action_heading_weight': args.action_heading_weight,
            'goal_heading_weight': 1.0 - args.action_heading_weight,
            'cruise_speed_mps': args.cruise_speed,
            'near_human_speed_mps': args.near_human_speed,
            'heading_gain': args.heading_gain,
            'omega_max_radps': args.omega_max,
        },
        'note': (
            'This is a one-step successor-ranking analysis. It does not model wheel slip, '
            'motor dynamics, odometry error, or the independent laser safety layer.'
        ),
    }
    output_path, csv_path = write_results(
        args, rows, summary, outcome_summary, metadata
    )
    aggregate = summary['aggregate']
    print('\n[AGGREGATE]')
    print(f"  samples:                    {aggregate['sample_count']}")
    print(
        f"  raw argmax agreement:       {100.0 * aggregate['raw_argmax_agreement']:.2f}%"
    )
    print(
        f"  decision argmax agreement:  {100.0 * aggregate['decision_argmax_agreement']:.2f}%"
    )
    print(f"  nominal scoring parity:      {100.0 * aggregate['nominal_parity']:.2f}%")
    print(f"  mean top-5 overlap:         {100.0 * aggregate['mean_top5_overlap']:.2f}%")
    print(f"  mean differential regret:   {aggregate['mean_diffdrive_regret']:.6f}")
    print(f"  p95 differential regret:    {aggregate['p95_diffdrive_regret']:.6f}")
    print(f'[INFO] JSON: {output_path}')
    print(f'[INFO] per-state CSV: {csv_path}')


if __name__ == '__main__':
    main()
