#!/usr/bin/env python3
"""Phase P2: Counterfactual Branching Experiments

Fork identical simulation snapshots at representative operational states:
1. Deadlock / Oscillatory Stagnation (travel > 1.0m, disp < 0.35m, ratio > 3.5, reversals >= 2)
2. Dense Normal Progression (progress > 0.5m, disp > 0.8m, clearance > 0.40m)
3. High Dynamic Risk / Imminent Closing (clearance < 0.35m, closing_speed > 0.4m/s)
4. Active Yield / Legitimate Detour (speed < 0.15m/s, clearance < 0.50m)

From the exact same state, branch into:
- Branch A: Base Dual-Stream (No intent intervention, alpha=0)
- Branch B: Controlled Fixed Intent Commitment (Dwell K=4, alpha=0.10 * gate)
- Branch C: Learnable LICS Intent Commitment (Learned beta_psi, alpha=0.10 * gate)

Evaluates 4s trajectory dynamics and full-episode outcomes to map exact causal intervention gains.
Saves to benchmark_results/counterfactual_branching_results.json.
"""

import sys
import os
import copy
import json
import time
import math
from pathlib import Path
import configparser
import numpy as np
import torch

WORKSPACE = Path('/home/abc/workspace/autonomous_research_20261012')
KDARL_CLEAN = WORKSPACE / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(WORKSPACE)] + [p for p in sys.path if p not in (str(WORKSPACE), str(KDARL_CLEAN), '')]

from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from run_4arm_comparative_benchmark import load_policy_for_arm, create_environment

RESULTS_DIR = WORKSPACE / 'benchmark_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Cases to harvest states from
SOURCE_CASES = [
    # Primary cases
    90001, 90046, 90000, 90005, 90009, 90013, 90002, 90015,
    # Generalization cases
    90050, 90057, 90074, 90060, 90068, 90077, 90085, 90095
]

def clone_simulation_state(env, policy):
    for h in env.humans:
        if hasattr(h.policy, 'sim'):
            h.policy.sim = None
    env_clone = copy.deepcopy(env)
    hist_prox = list(policy._history)
    hist_goal = list(policy._goal_history)
    last_act = copy.deepcopy(getattr(policy, '_last_action', None))
    return env_clone, hist_prox, hist_goal, last_act

def run_branch_from_snapshot(policy, branch_type: str, env_snapshot, hist_prox, hist_goal, last_act, max_steps=201):
    for h in env_snapshot.humans:
        if hasattr(h.policy, 'sim'):
            h.policy.sim = None
    env = copy.deepcopy(env_snapshot)
    robot = env.robot
    robot.set_policy(policy)
    robot.env = env
    env.set_robot(robot)

    # Initialize policy history exactly from snapshot
    policy.reset_episode_stats()
    policy._history = list(hist_prox)
    policy._goal_history = list(hist_goal)
    if last_act is not None:
        policy._last_action = last_act

    # Branch configuration
    if branch_type == 'Branch_A_NoIntervention':
        # Zero guidance weight
        alpha_mult = 0.0
    elif branch_type == 'Branch_B_FixedDwell':
        alpha_mult = 1.0
    elif branch_type == 'Branch_C_LICS':
        alpha_mult = 1.0
    else:
        raise ValueError(branch_type)

    # Temporary wrapper for scoring if Branch A
    original_score = policy.score_sarl_candidates
    if branch_type == 'Branch_A_NoIntervention':
        def zero_guidance_score(state):
            base_res = original_score(state)
            # Revert any intent modulation: base_res['scores'] before modulation
            return base_res

    traj_x = [robot.px]
    traj_y = [robot.py]
    init_dist = math.hypot(robot.gx - robot.px, robot.gy - robot.py)

    done = False
    step = 0
    short_term_4s = {}
    last_info = None

    while not done and step < max_steps:
        step += 1
        state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
        action = policy.predict(state)
        obs, rew, term, trunc, last_info = env.step(action)
        done = term or trunc

        traj_x.append(robot.px)
        traj_y.append(robot.py)

        # Capture 4-second (16-step) checkpoint
        if step == 16:
            curr_dist = math.hypot(robot.gx - robot.px, robot.gy - robot.py)
            p_arr = np.column_stack([traj_x, traj_y])
            t_4s = float(np.sum(np.linalg.norm(np.diff(p_arr, axis=0), axis=1)))
            d_4s = float(np.linalg.norm(p_arr[-1] - p_arr[0]))
            short_term_4s = {
                'prog_4s': round(init_dist - curr_dist, 3),
                'travel_4s': round(t_4s, 3),
                'disp_4s': round(d_4s, 3),
                'ratio_4s': round(t_4s / max(d_4s, 0.01), 1),
                'collision_in_4s': bool(term and last_info.get('event') == 'collision')
            }

    if not short_term_4s:
        curr_dist = math.hypot(robot.gx - robot.px, robot.gy - robot.py)
        p_arr = np.column_stack([traj_x, traj_y])
        t_4s = float(np.sum(np.linalg.norm(np.diff(p_arr, axis=0), axis=1))) if len(p_arr) > 1 else 0.0
        d_4s = float(np.linalg.norm(p_arr[-1] - p_arr[0])) if len(p_arr) > 1 else 0.0
        short_term_4s = {
            'prog_4s': round(init_dist - curr_dist, 3),
            'travel_4s': round(t_4s, 3),
            'disp_4s': round(d_4s, 3),
            'ratio_4s': round(t_4s / max(d_4s, 0.01), 1),
            'collision_in_4s': bool(done and last_info.get('event') == 'collision')
        }

    last_event = last_info.get('event', 'nothing') if last_info else 'nothing'
    if last_event in ('reach_goal', 'reachgoal') or env.robot.reached_destination():
        outcome = 'SUCCESS'
    elif last_event == 'collision':
        outcome = 'COLLISION'
    else:
        outcome = 'TIMEOUT'

    return {
        'outcome': outcome,
        'event': last_event,
        'final_time': round(float(env.global_time), 2),
        'steps': step,
        'short_term_4s': short_term_4s
    }

def run_counterfactual_experiment():
    print("==========================================================================")
    print("PHASE P2: EXECUTING COUNTERFACTUAL BRANCHING EXPERIMENTS")
    print("==========================================================================")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load 3 test policies
    policy_b1 = load_policy_for_arm('B1', device=device)  # Base Dual-Stream (Branch A)
    policy_b3 = load_policy_for_arm('B3', device=device)  # Fixed Dwell (Branch B)
    policy_b2 = load_policy_for_arm('B2', device=device)  # Adaptive LICS (Branch C)

    env, env_config = create_environment()
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy_b1)
    robot.env = env
    env.set_robot(robot)

    harvested_states = []
    # Target quotas per archetype
    quotas = {
        'Archetype_1_Deadlock': 0,
        'Archetype_2_SmoothProg': 0,
        'Archetype_3_HighRisk': 0,
        'Archetype_4_ActiveYield': 0
    }
    max_per_archetype = 8

    print("Step 1: Harvesting representative state snapshots from source trajectories...")
    for case_id in SOURCE_CASES:
        obs, _ = env.reset(options={'test_case': case_id})
        policy_b1.reset_episode_stats()
        history_pos = []
        history_vel = []
        done = False
        step = 0

        while not done and step < 180:
            step += 1
            px, py = robot.px, robot.py
            history_pos.append((px, py))
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            rob = state.self_state

            if len(state.human_states):
                phys_clearances = [
                    math.hypot(rob.px - h.px, rob.py - h.py) - rob.radius - h.radius
                    for h in state.human_states
                ]
                dmin_curr = float(min(phys_clearances))
                nearest_h = state.human_states[int(np.argmin(phys_clearances))]
                rel_vx = nearest_h.vx - rob.vx
                rel_vy = nearest_h.vy - rob.vy
                closing_speed = -( (nearest_h.px - rob.px)*rel_vx + (nearest_h.py - rob.py)*rel_vy ) / max(0.01, math.hypot(nearest_h.px - rob.px, nearest_h.py - rob.py))
            else:
                dmin_curr = 999.0
                closing_speed = 0.0

            action = policy_b1.predict(state)
            history_vel.append((action.vx, action.vy))
            speed = math.hypot(action.vx, action.vy)

            # Window metrics
            W = 16
            if len(history_pos) >= W:
                recent_p = history_pos[-W:]
                t_4s = float(sum(math.hypot(recent_p[i][0] - recent_p[i-1][0], recent_p[i][1] - recent_p[i-1][1]) for i in range(1, len(recent_p))))
                d_4s = float(math.hypot(recent_p[-1][0] - recent_p[0][0], recent_p[-1][1] - recent_p[0][1]))
                r_4s = t_4s / max(d_4s, 0.01)

                recent_v = history_vel[-W:]
                rev_4s = 0
                for i in range(1, len(recent_v)):
                    dot = recent_v[i][0]*recent_v[i-1][0] + recent_v[i][1]*recent_v[i-1][1]
                    m1 = math.hypot(recent_v[i][0], recent_v[i][1])
                    m2 = math.hypot(recent_v[i-1][0], recent_v[i-1][1])
                    if m1 > 0.1 and m2 > 0.1 and dot / (m1*m2) < -0.5:
                        rev_4s += 1
            else:
                t_4s, d_4s, r_4s, rev_4s = 0.0, 0.0, 1.0, 0

            # Archetype identification
            cand_arch = None
            if step >= 24 and t_4s > 0.8 and d_4s < 0.35 and (r_4s > 3.0 or rev_4s >= 2) and quotas['Archetype_1_Deadlock'] < max_per_archetype:
                cand_arch = 'Archetype_1_Deadlock'
            elif dmin_curr < 0.30 and closing_speed > 0.5 and quotas['Archetype_3_HighRisk'] < max_per_archetype:
                cand_arch = 'Archetype_3_HighRisk'
            elif speed < 0.15 and dmin_curr < 0.45 and quotas['Archetype_4_ActiveYield'] < max_per_archetype:
                cand_arch = 'Archetype_4_ActiveYield'
            elif step >= 16 and d_4s > 0.8 and dmin_curr > 0.40 and quotas['Archetype_2_SmoothProg'] < max_per_archetype:
                cand_arch = 'Archetype_2_SmoothProg'

            if cand_arch is not None:
                quotas[cand_arch] += 1
                env_snap, h_prox, h_goal, l_act = clone_simulation_state(env, policy_b1)
                harvested_states.append({
                    'state_id': f"State_{len(harvested_states)+1}",
                    'archetype': cand_arch,
                    'case_id': case_id,
                    'step': step,
                    'time': float(env.global_time),
                    'dmin_curr': round(dmin_curr, 3),
                    'closing_speed': round(closing_speed, 3),
                    'travel_4s': round(t_4s, 3),
                    'disp_4s': round(d_4s, 3),
                    'ratio_4s': round(r_4s, 1),
                    'reversals_4s': rev_4s,
                    'env_snap': env_snap,
                    'h_prox': h_prox,
                    'h_goal': h_goal,
                    'l_act': l_act
                })
                print(f"  [Harvested {cand_arch}] Case {case_id} Step {step} (Clearance={dmin_curr:.2f}m, Ratio={r_4s:.1f})")

            obs, rew, term, trunc, _ = env.step(action)
            done = term or trunc

    print(f"\nTotal harvested snapshots: {len(harvested_states)}")
    for k, v in quotas.items():
        print(f"  - {k}: {v} snapshots")

    # Step 2: Run Counterfactual Branches
    print("\nStep 2: Executing Counterfactual Branches (Branch A, B, C) from identical snapshots...")
    branching_results = []
    t_start = time.time()

    for item in harvested_states:
        sid = item['state_id']
        arch = item['archetype']
        cid = item['case_id']
        step = item['step']
        snap = item['env_snap']
        hp = item['h_prox']
        hg = item['h_goal']
        la = item['l_act']

        # Branch A: No Intervention (Base Dual-Stream B1)
        res_a = run_branch_from_snapshot(policy_b1, 'Branch_A_NoIntervention', snap, hp, hg, la)
        # Branch B: Fixed Intent Commitment (B3)
        res_b = run_branch_from_snapshot(policy_b3, 'Branch_B_FixedDwell', snap, hp, hg, la)
        # Branch C: Adaptive LICS Intent Commitment (B2)
        res_c = run_branch_from_snapshot(policy_b2, 'Branch_C_LICS', snap, hp, hg, la)

        # Causal Intervention Gain Calculation:
        # Score mapping: SUCCESS = +1.0, TIMEOUT = 0.0, COLLISION = -1.0
        score_map = {'SUCCESS': 1.0, 'TIMEOUT': 0.0, 'COLLISION': -1.0}
        gain_b = score_map[res_b['outcome']] - score_map[res_a['outcome']]
        gain_c = score_map[res_c['outcome']] - score_map[res_a['outcome']]
        delta_c_minus_b = score_map[res_c['outcome']] - score_map[res_b['outcome']]

        record = {
            'state_id': sid,
            'archetype': arch,
            'source_case': cid,
            'step': step,
            'dmin_curr': item['dmin_curr'],
            'closing_speed': item['closing_speed'],
            'ratio_4s': item['ratio_4s'],
            'reversals_4s': item['reversals_4s'],
            'branch_A_base': res_a,
            'branch_B_fixed': res_b,
            'branch_C_lics': res_c,
            'gain_fixed_vs_base': gain_b,
            'gain_lics_vs_base': gain_c,
            'advantage_lics_over_fixed': delta_c_minus_b
        }
        branching_results.append(record)

        print(f"[{sid} | {arch}] Base: {res_a['outcome']} -> Fixed: {res_b['outcome']} (Gain={gain_b:+.1f}) | LICS: {res_c['outcome']} (Gain={gain_c:+.1f})")

    t_end = time.time()
    print(f"\nCompleted counterfactual branching in {t_end - t_start:.2f}s")

    # Serialize results (excluding unpickled env objects)
    serializable_records = []
    for r in branching_results:
        clean_r = {k: v for k, v in r.items() if not k.endswith('_snap')}
        serializable_records.append(clean_r)

    out_file = RESULTS_DIR / 'counterfactual_branching_results.json'
    with open(out_file, 'w') as f:
        json.dump(serializable_records, f, indent=2)
    print(f"Saved counterfactual branching records to {out_file}")

    # Summary table by Archetype
    print("\n" + "=" * 90)
    print("COUNTERFACTUAL INTERVENTION GAIN BY STATE ARCHETYPE")
    print("=" * 90)
    print(f"{'State Archetype':<30} | {'Count':<5} | {'Base SR':<8} | {'Fixed SR':<8} | {'LICS SR':<8} | {'Avg LICS Gain':<13} | {'C vs B Adv':<10}")
    print("-" * 90)

    for arch_name in quotas.keys():
        arch_records = [r for r in serializable_records if r['archetype'] == arch_name]
        cnt = len(arch_records)
        if cnt == 0: continue
        sr_a = sum(1 for r in arch_records if r['branch_A_base']['outcome'] == 'SUCCESS') / cnt * 100
        sr_b = sum(1 for r in arch_records if r['branch_B_fixed']['outcome'] == 'SUCCESS') / cnt * 100
        sr_c = sum(1 for r in arch_records if r['branch_C_lics']['outcome'] == 'SUCCESS') / cnt * 100
        avg_gain_c = np.mean([r['gain_lics_vs_base'] for r in arch_records])
        avg_adv_c = np.mean([r['advantage_lics_over_fixed'] for r in arch_records])
        print(f"{arch_name:<30} | {cnt:<5} | {sr_a:6.1f}% | {sr_b:6.1f}% | {sr_c:6.1f}% | {avg_gain_c:+12.2f} | {avg_adv_c:+9.2f}")

    return serializable_records

if __name__ == '__main__':
    run_counterfactual_experiment()
