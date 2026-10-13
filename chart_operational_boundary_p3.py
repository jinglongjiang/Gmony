#!/usr/bin/env python3
"""Phase P3: Charting Operational Boundary & Independent Validation (100 Fresh Cases)

1. Analyzes counterfactual branching records from benchmark_results/counterfactual_branching_results.json.
2. Constructs a 2D Operational Boundary Grid along Stagnation Severity and Collision Risk.
3. Formulates the Boundary-Gated Policy:
   - Active ONLY in Region A (High Deadlock, Moderate/Safe Clearance).
   - Fallback to Base Reactive Lookahead in Region B (High Collision Risk) and Region C (Smooth Progress).
4. Tests Base vs Unconditional LICS vs Boundary-Gated LICS on 100 Fresh Cases (Cases 90150 - 90249).
5. Outputs definitive statistical verification.
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
from strat_mamba_dual_stream import StratMambaDualStreamPolicy
from learnable_intent_mamba import LICSStratMambaPolicy, HeuristicHysteresisPolicy

CHECKPOINT_DIR = WORKSPACE / 'checkpoints'
RESULTS_DIR = WORKSPACE / 'benchmark_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_p2_counterfactuals():
    p2_file = RESULTS_DIR / 'counterfactual_branching_results.json'
    if not p2_file.exists():
        print(f"Error: {p2_file} not found! Run P2 first.")
        return None

    with open(p2_file) as f:
        records = json.load(f)

    print("\n==========================================================================")
    print(f"PHASE P3: ANALYZING {len(records)} COUNTERFACTUAL BRANCHING EXPERIMENTS")
    print("==========================================================================")

    # 1. Breakdown by Archetype
    archetypes = sorted(list(set(r['archetype'] for r in records)))
    summary_by_arch = {}

    for arch in archetypes:
        arch_recs = [r for r in records if r['archetype'] == arch]
        n = len(arch_recs)
        succ_a = sum(1 for r in arch_recs if r['branch_A_base']['outcome'] == 'SUCCESS')
        succ_b = sum(1 for r in arch_recs if r['branch_B_fixed']['outcome'] == 'SUCCESS')
        succ_c = sum(1 for r in arch_recs if r['branch_C_lics']['outcome'] == 'SUCCESS')

        coll_a = sum(1 for r in arch_recs if r['branch_A_base']['outcome'] == 'COLLISION')
        coll_b = sum(1 for r in arch_recs if r['branch_B_fixed']['outcome'] == 'COLLISION')
        coll_c = sum(1 for r in arch_recs if r['branch_C_lics']['outcome'] == 'COLLISION')

        avg_gain_b = float(np.mean([r['gain_fixed_vs_base'] for r in arch_recs]))
        avg_gain_c = float(np.mean([r['gain_lics_vs_base'] for r in arch_recs]))
        adv_c_over_b = float(np.mean([r['advantage_lics_over_fixed'] for r in arch_recs]))

        summary_by_arch[arch] = {
            'n': n,
            'sr_base': round(succ_a / n * 100, 1),
            'sr_fixed': round(succ_b / n * 100, 1),
            'sr_lics': round(succ_c / n * 100, 1),
            'cr_base': round(coll_a / n * 100, 1),
            'cr_fixed': round(coll_b / n * 100, 1),
            'cr_lics': round(coll_c / n * 100, 1),
            'mean_gain_fixed': round(avg_gain_b, 2),
            'mean_gain_lics': round(avg_gain_c, 2),
            'adv_lics_over_fixed': round(adv_c_over_b, 2)
        }

    # Print Archetype Breakdown Table
    print(f"\n{'State Archetype':<28} | {'N':<3} | {'Base SR':<8} | {'Fixed SR':<8} | {'LICS SR':<8} | {'Base CR':<8} | {'LICS CR':<8} | {'LICS Gain':<10} | {'LICS vs Fix':<11}")
    print("-" * 110)
    for arch, s in summary_by_arch.items():
        print(f"{arch:<28} | {s['n']:<3} | {s['sr_base']:6.1f}% | {s['sr_fixed']:6.1f}% | {s['sr_lics']:6.1f}% | {s['cr_base']:6.1f}% | {s['cr_lics']:6.1f}% | {s['mean_gain_lics']:+9.2f} | {s['adv_lics_over_fixed']:+10.2f}")

    # 2. 2D Operational Boundary Map (Stagnation Ratio vs Current Clearance)
    print("\n" + "=" * 90)
    print("2D OPERATIONAL REGIME MAP: GAIN OF INTENT INTERVENTION [DELTA = Score_LICS - Score_Base]")
    print("=" * 90)

    # Grid bins
    ratio_bins = [('Low (<=2.0)', 0.0, 2.0), ('Med (2.0-3.5)', 2.0, 3.5), ('High (>3.5)', 3.5, 999.0)]
    dmin_bins = [('Safe (>0.40m)', 0.40, 999.0), ('Mod (0.28-0.40m)', 0.28, 0.40), ('Tight (<0.28m)', -999.0, 0.28)]

    grid_matrix = {}
    header_col = "Clearance / Stagnation"
    print(f"{header_col:<25} | {'Low Ratio (<=2.0)':<20} | {'Med Ratio (2.0-3.5)':<20} | {'High Deadlock (>3.5)':<20}")
    print("-" * 90)

    for d_name, d_low, d_high in dmin_bins:
        row_str = f"{d_name:<25} | "
        for r_name, r_low, r_high in ratio_bins:
            cell_recs = [
                r for r in records
                if d_low <= r['dmin_curr'] < d_high and r_low <= r['ratio_4s'] < r_high
            ]
            if len(cell_recs) > 0:
                avg_g = np.mean([r['gain_lics_vs_base'] for r in cell_recs])
                cell_label = f"{avg_g:+5.2f} (n={len(cell_recs)})"
            else:
                cell_label = "  N/A  (n=0) "
            row_str += f"{cell_label:<20} | "
        print(row_str)

    return summary_by_arch

class BoundaryGatedPolicy(LICSStratMambaPolicy):
    """Event/State-Conditioned Meta Policy with LICS:
    Operates in Pure Base Reactive Mode by default (guaranteeing low collision rate).
    Engages Intent Commitment ONLY when verified to be in Region A:
    - High Stagnation (ratio_4s > 3.0 or rev_4s >= 2 with travel_4s > 0.5 & disp_4s < 0.35)
      OR Yielding Stall (travel_4s < 0.35 & disp_4s < 0.25 after step >= 20)
    - Moderate Clearance (0.22m <= current_physical_clearance < 0.60m)
    Exits immediately when deadlock dissolves or clearance drops.
    """
    def __init__(self, config=None, device='cuda', backbone='gru', hysteresis_margin=0.15):
        super().__init__(config=config, device=device, backbone=backbone, hysteresis_margin=hysteresis_margin)
        self.history_pos = []
        self.history_vel = []

    def reset_episode_stats(self):
        super().reset_episode_stats()
        self.history_pos.clear()
        self.history_vel.clear()

    def score_sarl_candidates(self, state: JointState):
        rob = state.self_state
        self.history_pos.append((rob.px, rob.py))

        # 1. Base scoring
        base_res = super(LICSStratMambaPolicy, self).score_sarl_candidates(state)

        # 2. Compute current physical clearance
        if len(state.human_states):
            curr_phys_dists = [
                math.hypot(rob.px - h.px, rob.py - h.py) - rob.radius - h.radius
                for h in state.human_states
            ]
            current_physical_dmin = float(min(curr_phys_dists))
        else:
            current_physical_dmin = 999.0

        # 3. Calculate 4-second trajectory metrics (W=16 steps)
        W = 16
        if len(self.history_pos) >= W:
            recent_p = self.history_pos[-W:]
            t_4s = float(sum(math.hypot(recent_p[i][0] - recent_p[i-1][0], recent_p[i][1] - recent_p[i-1][1]) for i in range(1, len(recent_p))))
            d_4s = float(math.hypot(recent_p[-1][0] - recent_p[0][0], recent_p[-1][1] - recent_p[0][1]))
            r_4s = t_4s / max(d_4s, 0.01)

            recent_v = self.history_vel[-W:]
            rev_4s = 0
            for i in range(1, len(recent_v)):
                dot = recent_v[i][0]*recent_v[i-1][0] + recent_v[i][1]*recent_v[i-1][1]
                m1 = math.hypot(recent_v[i][0], recent_v[i][1])
                m2 = math.hypot(recent_v[i-1][0], recent_v[i-1][1])
                if m1 > 0.1 and m2 > 0.1 and dot / (m1*m2) < -0.5:
                    rev_4s += 1
        else:
            t_4s, d_4s, r_4s, rev_4s = 0.0, 0.0, 1.0, 0

        # 4. Region A Boundary Condition:
        is_oscillatory = (t_4s > 0.5 and d_4s < 0.35 and (r_4s > 3.0 or rev_4s >= 2))
        is_yield_stall = (t_4s < 0.35 and d_4s < 0.25 and len(self.history_pos) >= 20)
        is_in_region_a = (len(self.history_pos) >= 16 and (is_oscillatory or is_yield_stall) and (0.22 <= current_physical_dmin < 0.60))

        if is_in_region_a:
            # Engage LICS Intent Guidance
            base_hist = list(self._history) + [base_res['current_token']]
            base_goal_hist = list(self._goal_history) + [base_res['current_goal']]
            h_fused = self.get_current_hidden_state(base_hist, base_goal_hist)
            intent, beta, switched = self.select_or_update_intent(h_fused, rob, current_physical_dmin)
            compat = self.intent_head.compute_action_intent_compatibility(base_res['commands'], rob, intent).to(self.device)
            gate = max(0.0, min(1.0, (current_physical_dmin - 0.22) / 0.15))
            alpha = 0.10 * gate
            modulated_scores = base_res['scores'] + alpha * compat
        else:
            # Stay in Pure Base Reactive Mode (alpha = 0)
            modulated_scores = base_res['scores']
            intent, beta, switched = -1, 1.0, False
            self.active_intent = None
            self.intent_dwell_steps = 0

        # Apply strict safety filter
        if self._phase in ('test', 'val', 'eval') and self.test_min_clearance > 0.0:
            safe_mask = base_res['clearance'] >= self.test_min_clearance
            if safe_mask.any():
                modulated_scores = modulated_scores.masked_fill(~safe_mask, -1e9)

        base_res['scores'] = modulated_scores
        base_res['active_intent'] = intent
        base_res['termination_prob'] = beta
        base_res['switched'] = switched
        return base_res

    def predict(self, state):
        act = super().predict(state)
        self.history_vel.append((act.vx, act.vy))
        return act

class BoundaryGatedFixedDwellPolicy(HeuristicHysteresisPolicy):
    """Event/State-Conditioned Meta Policy with Fixed Dwell:
    Operates in Pure Base Reactive Mode by default.
    Engages Fixed Dwell (K=4) ONLY when verified to be in Region A.
    """
    def __init__(self, config=None, device='cuda', backbone='gru', dwell_k=4):
        super().__init__(config=config, device=device, backbone=backbone, dwell_k=dwell_k)
        self.history_pos = []
        self.history_vel = []

    def reset_episode_stats(self):
        super().reset_episode_stats()
        self.history_pos.clear()
        self.history_vel.clear()

    def score_sarl_candidates(self, state: JointState):
        rob = state.self_state
        self.history_pos.append((rob.px, rob.py))

        if len(state.human_states):
            curr_phys_dists = [
                math.hypot(rob.px - h.px, rob.py - h.py) - rob.radius - h.radius
                for h in state.human_states
            ]
            current_physical_dmin = float(min(curr_phys_dists))
        else:
            current_physical_dmin = 999.0

        W = 16
        if len(self.history_pos) >= W:
            recent_p = self.history_pos[-W:]
            t_4s = float(sum(math.hypot(recent_p[i][0] - recent_p[i-1][0], recent_p[i][1] - recent_p[i-1][1]) for i in range(1, len(recent_p))))
            d_4s = float(math.hypot(recent_p[-1][0] - recent_p[0][0], recent_p[-1][1] - recent_p[0][1]))
            r_4s = t_4s / max(d_4s, 0.01)

            recent_v = self.history_vel[-W:]
            rev_4s = 0
            for i in range(1, len(recent_v)):
                dot = recent_v[i][0]*recent_v[i-1][0] + recent_v[i][1]*recent_v[i-1][1]
                m1 = math.hypot(recent_v[i][0], recent_v[i][1])
                m2 = math.hypot(recent_v[i-1][0], recent_v[i-1][1])
                if m1 > 0.1 and m2 > 0.1 and dot / (m1*m2) < -0.5:
                    rev_4s += 1
        else:
            t_4s, d_4s, r_4s, rev_4s = 0.0, 0.0, 1.0, 0

        is_oscillatory = (t_4s > 0.5 and d_4s < 0.35 and (r_4s > 3.0 or rev_4s >= 2))
        is_yield_stall = (t_4s < 0.35 and d_4s < 0.25 and len(self.history_pos) >= 20)
        is_in_region_a = (len(self.history_pos) >= 16 and (is_oscillatory or is_yield_stall) and (0.22 <= current_physical_dmin < 0.60))

        if is_in_region_a or self.steps_remaining > 0:
            return super().score_sarl_candidates(state)
        else:
            self.committed_intent = None
            self.steps_remaining = 0
            return super(HeuristicHysteresisPolicy, self).score_sarl_candidates(state)

    def predict(self, state):
        act = super().predict(state)
        self.history_vel.append((act.vx, act.vy))
        return act

def run_independent_validation(num_cases=100, start_case=90150, device='cuda'):
    print("\n==========================================================================")
    print(f"STEP 2: INDEPENDENT VALIDATION ON {num_cases} FRESH CASES (Cases {start_case} - {start_case + num_cases - 1})")
    print("==========================================================================")

    from run_4arm_comparative_benchmark import get_base_config, CHECKPOINT_DIR
    from run_selection_study_50cases import install_selection
    cfg = get_base_config()
    weights_path = WORKSPACE / 'baseline_weights/gru-t24-seed443.pth'
    b2_ckpt_path = CHECKPOINT_DIR / 'b2_lics_trained.pth'

    # 1. Base Dual-Stream (B1)
    pol_b1 = load_policy_for_arm('B1', device=device)

    # 2. Fixed Dwell-4 Baseline (B3)
    pol_b3 = load_policy_for_arm('B3', device=device)

    # 3. Unconditional LICS (B2)
    pol_b2 = load_policy_for_arm('B2', device=device)

    # 4. Boundary-Gated LICS
    pol_gated_lics = BoundaryGatedPolicy(cfg, device=device, backbone='gru')
    if b2_ckpt_path.exists():
        pol_gated_lics.load_state_dict(torch.load(b2_ckpt_path, map_location=device), strict=False)
    else:
        pol_gated_lics.load_pretrained_weights(weights_path)
    pol_gated_lics.eval()
    pol_gated_lics.build_action_space(1.0)
    pol_gated_lics.use_sarl_predict = True
    pol_gated_lics.set_phase('test')
    pol_gated_lics.test_min_clearance = 0.2
    pol_gated_lics.test_risk_lambda = 0.8
    pol_gated_lics.test_action_smoothing = 0.3
    install_selection(pol_gated_lics, 'ttc5')

    # 5. Boundary-Gated Fixed Dwell-4
    pol_gated_dwell = BoundaryGatedFixedDwellPolicy(cfg, device=device, backbone='gru', dwell_k=4)
    pol_gated_dwell.load_pretrained_weights(weights_path)
    pol_gated_dwell.eval()
    pol_gated_dwell.build_action_space(1.0)
    pol_gated_dwell.use_sarl_predict = True
    pol_gated_dwell.set_phase('test')
    pol_gated_dwell.test_min_clearance = 0.2
    pol_gated_dwell.test_risk_lambda = 0.8
    pol_gated_dwell.test_action_smoothing = 0.3
    install_selection(pol_gated_dwell, 'ttc5')

    eval_policies = [
        ('Base_DualStream_B1', pol_b1),
        ('Fixed_Dwell4_B3', pol_b3),
        ('Unconditional_LICS_B2', pol_b2),
        ('BoundaryGated_LICS', pol_gated_lics),
        ('BoundaryGated_FixedDwell', pol_gated_dwell)
    ]

    val_results = {}
    for pol_name, pol in eval_policies:
        print(f"\nEvaluating {pol_name} on {num_cases} fresh cases...")
        env, env_config = create_environment()
        robot = Robot(env_config, 'robot')
        robot.set_policy(pol)
        robot.env = env
        env.set_robot(robot)

        success = 0
        collision = 0
        timeout = 0
        stagnation = 0
        times = []
        case_records = []

        t0 = time.time()
        for i in range(num_cases):
            case_id = start_case + i
            obs, _ = env.reset(options={'test_case': case_id})
            pol.reset_episode_stats()
            traj_x, traj_y = [], []
            min_dist = 999.0
            done = False
            step = 0
            last_info = None

            while not done and step < 201:
                step += 1
                state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
                if len(env.humans):
                    dists = [math.hypot(robot.px - h.px, robot.py - h.py) - robot.radius - h.radius for h in env.humans]
                    min_dist = min(min_dist, min(dists))
                action = pol.predict(state)
                traj_x.append(robot.px)
                traj_y.append(robot.py)
                obs, rew, term, trunc, last_info = env.step(action)
                done = term or trunc

            last_event = last_info.get('event', 'nothing') if last_info else 'nothing'
            if last_event == 'collision':
                outcome = 'COLLISION'
            elif last_event in ('reach_goal', 'reachgoal') or env.robot.reached_destination():
                outcome = 'SUCCESS'
                times.append(env.global_time)
            elif last_event in ('timeout', 'time_out') or env.global_time >= env.time_limit - 1e-4:
                outcome = 'TIMEOUT'
            else:
                outcome = 'TIMEOUT'

            is_stag = False
            if outcome == 'SUCCESS':
                success += 1
            elif outcome == 'COLLISION':
                collision += 1
            else:
                timeout += 1
                p_arr = np.column_stack([traj_x[-40:], traj_y[-40:]]) if len(traj_x) >= 40 else np.column_stack([traj_x, traj_y])
                t_10s = float(np.sum(np.linalg.norm(np.diff(p_arr, axis=0), axis=1))) if len(p_arr) > 1 else 0.0
                d_10s = float(np.linalg.norm(p_arr[-1] - p_arr[0])) if len(p_arr) > 1 else 0.0
                if t_10s > 1.5 and d_10s < 0.35 and t_10s / max(d_10s, 0.01) > 4.0:
                    is_stag = True
                    stagnation += 1

            case_records.append({
                'case_id': case_id,
                'outcome': outcome,
                'time': round(float(env.global_time), 2),
                'last_event': last_event,
                'stagnation': is_stag,
                'min_dist': round(float(min_dist), 3)
            })

        t1 = time.time()
        sr = success / num_cases * 100
        cr = collision / num_cases * 100
        tr = timeout / num_cases * 100
        sor = stagnation / num_cases * 100
        mean_t = float(np.mean(times)) if len(times) else 0.0

        val_results[pol_name] = {
            'sr': round(sr, 1),
            'cr': round(cr, 1),
            'tr': round(tr, 1),
            'sor': round(sor, 1),
            'mean_time': round(mean_t, 2),
            'elapsed_sec': round(t1 - t0, 1),
            'cases': case_records
        }
        print(f"[{pol_name}] SR: {sr:.1f}% ({success}/{num_cases}) | CR: {cr:.1f}% | TR: {tr:.1f}% | SOR: {sor:.1f}% | Time: {mean_t:.2f}s")

    out_file = RESULTS_DIR / 'independent_validation_100cases.json'
    with open(out_file, 'w') as f:
        json.dump(val_results, f, indent=2)
    print(f"\nSaved independent validation results to {out_file}")

    print("\n" + "=" * 80)
    print(f"INDEPENDENT VALIDATION SUMMARY (100 FRESH CASES: {start_case} - {start_case + num_cases - 1})")
    print("=" * 80)
    print(f"{'Policy Name':<28} | {'SR':<8} | {'CR':<8} | {'TR':<8} | {'SOR':<8} | {'Mean Time':<10}")
    print("-" * 80)
    for pol_name, r in val_results.items():
        print(f"{pol_name:<28} | {r['sr']:6.1f}% | {r['cr']:6.1f}% | {r['tr']:6.1f}% | {r['sor']:6.1f}% | {r['mean_time']:6.2f}s")

    return val_results

if __name__ == '__main__':
    analyze_p2_counterfactuals()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_independent_validation(num_cases=100, start_case=90150, device=device)
