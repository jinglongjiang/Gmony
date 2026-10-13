#!/usr/bin/env python3
"""Phase P1: State-Level Telemetry Reconstruction across Representative Scenarios

Reconstructs complete, high-resolution step-by-step trajectories across representative cases:
- Group 1: Stagnation / Deadlock Cases (Cases 90001, 90046, 90057, 90074)
- Group 2: Smooth Normal Progression Cases (Cases 90000, 90005, 90020, 90030)
- Group 3: High-Risk / Collision Cases (Cases 90009, 90013, 90022, 90034)
- Group 4: Normal Detour / Active Yield Cases (Cases 90002, 90015, 90025, 90040)

Captures full physics, clearance, value gap, intent, and behavioral taxonomy.
Saves to benchmark_results/state_level_telemetry.json.
"""

import sys
import os
import json
import time
import math
from pathlib import Path
import configparser
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parent
KDARL_CLEAN = WORKSPACE / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(WORKSPACE)] + [p for p in sys.path if p not in (str(WORKSPACE), str(KDARL_CLEAN), '')]

from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from run_4arm_comparative_benchmark import load_policy_for_arm, create_environment

RESULTS_DIR = WORKSPACE / 'benchmark_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIVE_CASES = [
    # Stagnation
    90001, 90046, 90057, 90074,
    # Smooth progress
    90000, 90005, 90020, 90030,
    # High risk / collision
    90009, 90013, 90022, 90034,
    # Detour / Yield
    90002, 90015, 90025, 90040
]

def collect_episode_telemetry(policy, case_id: int, policy_name: str, device='cuda'):
    env, env_config = create_environment()
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    robot.env = env
    env.set_robot(robot)

    obs, _ = env.reset(options={'test_case': case_id})
    policy.reset_episode_stats()

    history_pos = []
    history_vel = []
    history_dist = []
    steps_data = []

    gx, gy = robot.gx, robot.gy
    done = False
    step = 0
    last_info = None

    while not done and step < 201:
        step += 1
        px, py = robot.px, robot.py
        goal_dist = math.hypot(gx - px, gy - py)
        history_pos.append((px, py))
        history_dist.append(goal_dist)

        state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])

        # Step metrics
        rob = state.self_state
        if len(state.human_states):
            phys_clearances = [
                math.hypot(rob.px - h.px, rob.py - h.py) - rob.radius - h.radius
                for h in state.human_states
            ]
            current_physical_dmin = float(min(phys_clearances))
            nearest_idx = int(np.argmin(phys_clearances))
            nearest_h = state.human_states[nearest_idx]
            rel_vx = nearest_h.vx - rob.vx
            rel_vy = nearest_h.vy - rob.vy
            closing_speed = -( (nearest_h.px - rob.px)*rel_vx + (nearest_h.py - rob.py)*rel_vy ) / max(0.01, math.hypot(nearest_h.px - rob.px, nearest_h.py - rob.py))
        else:
            current_physical_dmin = 999.0
            closing_speed = 0.0

        # Run scoring
        res = policy.score_sarl_candidates(state)
        scores = res['scores'].cpu().numpy()
        cmds = res['commands']
        clearances = res['clearance'].cpu().numpy()

        # Count safe candidates (predicted 1-step clearance >= 0.20m)
        safe_count = int(np.sum(clearances >= policy.test_min_clearance))
        safe_ratio = safe_count / len(clearances)

        # Top 1 vs Top 2 value gap
        sorted_scores = np.sort(scores)[::-1]
        top1 = float(sorted_scores[0])
        top2 = float(sorted_scores[1]) if len(sorted_scores) > 1 else top1
        value_gap = top1 - top2

        # Intent telemetry (if LICS)
        active_intent = int(res.get('active_intent', -1))
        beta_prob = float(res.get('termination_prob', 0.0))
        switched = bool(res.get('switched', False))
        alpha = float(getattr(policy, 'intent_guidance_weight', torch.tensor(0.10)).item()) if hasattr(policy, 'intent_head') else 0.0

        action = policy.predict(state)
        history_vel.append((action.vx, action.vy))

        # Recent 4s window (16 steps) metrics
        W = 16
        if len(history_pos) >= 2:
            recent_p = history_pos[-min(len(history_pos), W):]
            travel_4s = float(sum(math.hypot(recent_p[i][0] - recent_p[i-1][0], recent_p[i][1] - recent_p[i-1][1]) for i in range(1, len(recent_p))))
            disp_4s = float(math.hypot(recent_p[-1][0] - recent_p[0][0], recent_p[-1][1] - recent_p[0][1]))
            ratio_4s = travel_4s / max(disp_4s, 0.01)
            progress_4s = float(history_dist[-min(len(history_dist), W)] - goal_dist)

            # Reversals
            recent_v = history_vel[-min(len(history_vel), W):]
            reversals_4s = 0
            for i in range(1, len(recent_v)):
                dot = recent_v[i][0]*recent_v[i-1][0] + recent_v[i][1]*recent_v[i-1][1]
                m1 = math.hypot(recent_v[i][0], recent_v[i][1])
                m2 = math.hypot(recent_v[i-1][0], recent_v[i-1][1])
                if m1 > 0.1 and m2 > 0.1 and dot / (m1*m2) < -0.5:
                    reversals_4s += 1
        else:
            travel_4s, disp_4s, ratio_4s, progress_4s, reversals_4s = 0.0, 0.0, 1.0, 0.0, 0

        # State behavioral classification
        curr_speed = math.hypot(action.vx, action.vy)
        if step >= 16 and travel_4s > 1.0 and disp_4s < 0.35 and (ratio_4s > 3.5 or reversals_4s >= 2):
            behavior = 'oscillatory_stagnation'
        elif curr_speed < 0.15 and current_physical_dmin < 0.45:
            behavior = 'active_yield'
        elif travel_4s > 0.8 and disp_4s > 0.5 and progress_4s < -0.1:
            behavior = 'normal_detour'
        else:
            behavior = 'normal_progression'

        step_record = {
            'step': step,
            'time': round(float(env.global_time), 2),
            'px': round(px, 3),
            'py': round(py, 3),
            'goal_dist': round(goal_dist, 3),
            'action_vx': round(float(action.vx), 3),
            'action_vy': round(float(action.vy), 3),
            'speed': round(curr_speed, 3),
            'travel_4s': round(travel_4s, 3),
            'disp_4s': round(disp_4s, 3),
            'ratio_4s': round(ratio_4s, 1),
            'progress_4s': round(progress_4s, 3),
            'reversals_4s': reversals_4s,
            'current_physical_dmin': round(current_physical_dmin, 3),
            'closing_speed': round(float(closing_speed), 3),
            'safe_candidates': safe_count,
            'safe_ratio': round(safe_ratio, 2),
            'value_top1': round(top1, 4),
            'value_top2': round(top2, 4),
            'value_gap': round(value_gap, 4),
            'active_intent': active_intent,
            'beta_prob': round(beta_prob, 3),
            'switched': switched,
            'behavior': behavior
        }
        steps_data.append(step_record)

        obs, reward, terminated, truncated, last_info = env.step(action)
        done = terminated or truncated

    last_event = last_info.get('event', 'nothing') if last_info else 'nothing'
    if last_event in ('reach_goal', 'reachgoal') or env.robot.reached_destination():
        outcome = 'SUCCESS'
    elif last_event == 'collision':
        outcome = 'COLLISION'
    else:
        outcome = 'TIMEOUT'

    return {
        'case_id': case_id,
        'policy_name': policy_name,
        'outcome': outcome,
        'event': last_event,
        'final_time': round(float(env.global_time), 2),
        'total_steps': step,
        'steps': steps_data
    }

def run_p1_collection():
    print("==========================================================================")
    print("PHASE P1: RECONSTRUCTING STATE-LEVEL TELEMETRY ACROSS REPRESENTATIVE CASES")
    print("==========================================================================")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    policies = {
        'B0': load_policy_for_arm('B0', device=device),
        'B1': load_policy_for_arm('B1', device=device),
        'B2': load_policy_for_arm('B2', device=device),
        'B3': load_policy_for_arm('B3', device=device)
    }

    telemetry_data = {}
    for case_id in REPRESENTATIVE_CASES:
        print(f"\nProcessing Case {case_id}...")
        telemetry_data[case_id] = {}
        for arm in ['B0', 'B1', 'B2']:
            ep_res = collect_episode_telemetry(policies[arm], case_id, arm, device=device)
            telemetry_data[case_id][arm] = ep_res
            print(f"  [{arm}] Outcome={ep_res['outcome']} (Event={ep_res['event']}), Steps={ep_res['total_steps']}, Time={ep_res['final_time']}s")

    out_file = RESULTS_DIR / 'state_level_telemetry.json'
    with open(out_file, 'w') as f:
        json.dump(telemetry_data, f, indent=2)
    print(f"\nSuccessfully saved detailed step-level telemetry to {out_file}")
    return telemetry_data

if __name__ == '__main__':
    run_p1_collection()
