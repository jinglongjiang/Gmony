#!/usr/bin/env python3
"""Stage 2: Targeted Harvesting of Deadlock and Harmful Interference Counterfactuals

Specifically targets:
1. True Deadlock States (where robot has been oscillating/wandering for 4s with low net displacement)
2. Negative Interference States (where robot was previously succeeding, but premature intent commitment causes collision)

Uses the audited pre-decision snapshot mechanism and 100% verified feature extraction.
Saves targeted counterfactuals to: benchmark_results/targeted_deadlock_and_harm_samples.json.
"""

import sys
import copy
import math
import json
import time
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
KDARL_CLEAN = REPO_ROOT / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(REPO_ROOT)] + [p for p in sys.path if p not in (str(REPO_ROOT), str(KDARL_CLEAN), '')]

from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from run_b0_counterfactual_study import (
    load_b0_policy, create_environment,
    clone_pre_decision_snapshot, run_branch,
    derive_macro_intent_from_action, INTENT_NAMES
)
from run_p3_cagis_benchmark import FEATURE_KEYS

RESULTS_DIR = REPO_ROOT / 'benchmark_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = RESULTS_DIR / 'targeted_deadlock_and_harm_samples.json'

def harvest_targeted_case(env, env_config, policy_orig, policy_branch, case_id, device='cuda'):
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy_orig)
    robot.env = env
    env.set_robot(robot)

    obs, _ = env.reset(options={'test_case': case_id})
    policy_orig.reset_episode_stats()

    history_pos = []
    history_vel = []
    case_snapshots = []
    done = False
    step = 0
    last_info = None

    while not done and step < 201:
        step += 1
        px, py = robot.px, robot.py
        history_pos.append((px, py))

        state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
        rob = state.self_state

        # Compute observable metrics
        if len(state.human_states):
            phys_clearances = [
                math.hypot(rob.px - h.px, rob.py - h.py) - rob.radius - h.radius
                for h in state.human_states
            ]
            dmin_curr = float(min(phys_clearances))
            nearest_h = state.human_states[int(np.argmin(phys_clearances))]
            rel_vx = nearest_h.vx - rob.vx
            rel_vy = nearest_h.vy - rob.vy
            closing_speed = -((nearest_h.px - rob.px)*rel_vx + (nearest_h.py - rob.py)*rel_vy) / max(0.01, math.hypot(nearest_h.px - rob.px, nearest_h.py - rob.py))
        else:
            dmin_curr = 999.0
            closing_speed = 0.0

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

        # Targeted sampling criteria:
        # Criterion 1: True Deadlock (sustained oscillation with stagnant net progress)
        is_deadlock = (step >= 24 and (r_4s >= 2.5 or rev_4s >= 2) and d_4s < 0.40 and t_4s > 0.5)
        
        # Criterion 2: Close Traffic Collision Risk (vulnerable state where intervention could either rescue or cause collision)
        is_vulnerable = (dmin_curr < 0.35 and closing_speed > 0.25 and 0.20 <= dmin_curr)

        cand_arch = None
        if is_deadlock:
            cand_arch = 'Deadlock_Targeted'
        elif is_vulnerable and step >= 10:
            cand_arch = 'Vulnerable_Interaction'

        # Harvest snapshot if candidate matches and not already sampled at this step
        if cand_arch is not None and len(case_snapshots) < 3:
            env_snap, hp, la = clone_pre_decision_snapshot(env, policy_orig)
            res_peek = policy_orig.score_sarl_candidates(state)
            best_cmd = res_peek['commands'][res_peek['scores'].argmax().item()]
            intent_candidate = derive_macro_intent_from_action(rob, best_cmd)

            dist_to_goal = math.hypot(rob.gx - rob.px, rob.gy - rob.py)
            theta_g = math.atan2(rob.gy - rob.py, rob.gx - rob.px)
            theta_v = math.atan2(rob.vy, rob.vx) if math.hypot(rob.vx, rob.vy) > 0.05 else theta_g
            goal_heading_error = (theta_v - theta_g + math.pi) % (2*math.pi) - math.pi
            safe_actions_ratio = float((res_peek['clearance'] >= 0.20).float().mean().item())

            case_snapshots.append({
                'snapshot_id': f"TargetSnap_C{case_id}_S{step}",
                'case_id': case_id,
                'step': step,
                'time': round(float(env.global_time), 2),
                'archetype': cand_arch,
                'features': {
                    'dmin_curr': round(dmin_curr, 3),
                    'closing_speed': round(closing_speed, 3),
                    'travel_4s': round(t_4s, 3),
                    'disp_4s': round(d_4s, 3),
                    'ratio_4s': round(r_4s, 2),
                    'reversals_4s': rev_4s,
                    'dist_to_goal': round(dist_to_goal, 2),
                    'goal_heading_error': round(goal_heading_error, 3),
                    'safe_actions_ratio': round(safe_actions_ratio, 3),
                    'current_speed': round(math.hypot(rob.vx, rob.vy), 3)
                },
                'intent_candidate': intent_candidate,
                'intent_name': INTENT_NAMES[intent_candidate],
                'env_snap': env_snap,
                'hp': hp,
                'la': la
            })

        action = policy_orig.predict(state)
        history_vel.append((action.vx, action.vy))
        obs, rew, term, trunc, last_info = env.step(action)
        done = term or trunc

    orig_event = last_info.get('event', 'nothing') if last_info else 'nothing'
    if orig_event == 'collision':
        orig_outcome = 'COLLISION'
    elif orig_event in ('reach_goal', 'reachgoal') or env.robot.reached_destination():
        orig_outcome = 'SUCCESS'
    else:
        orig_outcome = 'TIMEOUT'

    score_map = {'SUCCESS': 1.0, 'TIMEOUT': 0.0, 'COLLISION': -1.0}
    processed = []

    for snap in case_snapshots:
        s_step = snap['step']
        intent = snap['intent_candidate']
        esnap = snap['env_snap']
        hp = snap['hp']
        la = snap['la']

        # Branch A: K=0 (Base B0 Zero Intervention)
        res_a = run_branch(policy_branch, esnap, hp, la, s_step, dwell_k=0, committed_intent=None)
        
        # Branch C: K=4 (Medium Dwell)
        res_c = run_branch(policy_branch, esnap, hp, la, s_step, dwell_k=4, committed_intent=intent)

        gain_k4 = score_map[res_c['outcome']] - score_map[res_a['outcome']]
        time_saved = (res_a['final_time'] - res_c['final_time']) if (res_a['outcome'] == 'SUCCESS' and res_c['outcome'] == 'SUCCESS') else 0.0

        # Outcome categorizations
        is_rescue = bool(res_a['outcome'] != 'SUCCESS' and res_c['outcome'] == 'SUCCESS')
        is_harm = bool(res_a['outcome'] == 'SUCCESS' and res_c['outcome'] != 'SUCCESS')

        processed.append({
            'snapshot_id': snap['snapshot_id'],
            'case_id': case_id,
            'step': s_step,
            'time': snap['time'],
            'archetype': snap['archetype'],
            'features': snap['features'],
            'intent_candidate': snap['intent_candidate'],
            'intent_name': snap['intent_name'],
            'branch_A_base_k0': res_a,
            'branch_C_dwell_k4': res_c,
            'gain_vs_base': gain_k4,
            'time_saved': round(float(time_saved), 2),
            'is_rescue': is_rescue,
            'is_harm': is_harm
        })

    return processed

def main():
    print("=" * 85)
    print("STAGE 2: TARGETED HARVESTING OF DEADLOCK AND HARMFUL INTERVENTION SAMPLES")
    print("=" * 85)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env, env_config = create_environment()
    policy_orig = load_b0_policy(device=device)
    policy_branch = load_b0_policy(device=device)

    # Cases from training pool (Cases 90000 - 90199)
    # We focus on the timeout cases + dense cases
    with open(RESULTS_DIR / 'b0_counterfactual_study_200cases.json', 'r') as f:
        p1_data = json.load(f)

    timeout_cids = [c['case_id'] for c in p1_data if c['orig_outcome'] == 'TIMEOUT']
    success_cids = [c['case_id'] for c in p1_data if c['orig_outcome'] == 'SUCCESS'][:30]
    target_cids = sorted(list(set(timeout_cids + success_cids)))

    print(f"Scanning {len(target_cids)} targeted cases ({len(timeout_cids)} timeout cases + {len(success_cids)} success cases)...")

    all_harvested = []
    t_start = time.time()
    rescues = 0
    harms = 0

    for idx, cid in enumerate(target_cids):
        snaps = harvest_targeted_case(env, env_config, policy_orig, policy_branch, cid, device=device)
        for s in snaps:
            all_harvested.append(s)
            if s['is_rescue']: rescues += 1
            if s['is_harm']: harms += 1

        if (idx + 1) % 10 == 0 or idx == len(target_cids) - 1:
            print(f"Processed {idx+1}/{len(target_cids)} cases | Total snapshots: {len(all_harvested)} | Rescues: {rescues} | Harms: {harms} | Elapsed: {time.time()-t_start:.1f}s")

    with open(OUT_FILE, 'w') as f:
        json.dump(all_harvested, f, indent=2)

    print("\n" + "=" * 85)
    print(f"TARGETED HARVESTING COMPLETE: {len(all_harvested)} SNAPSHOTS HARVESTED IN {time.time()-t_start:.1f}s")
    print(f"  - Rescued Cases (Positive Gains): {rescues}")
    print(f"  - Harm-Inducing Cases (Negative/Collision Gains): {harms}")
    print(f"Saved dataset to: {OUT_FILE}")
    print("=" * 85)

if __name__ == '__main__':
    main()
