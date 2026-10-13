#!/usr/bin/env python3
"""Stage 4: Truly Independent 500-Case Closed-Loop Evaluation (Cases 90300 - 90799)

Rigorous Head-to-Head Comparison on 500 Completely Fresh, Unseen Cases:
- Arm 1: Pure B0 Baseline (Single-Stream GRU, Zero Intervention)
- Arm 2: Best Heuristic Gate (Hand-crafted threshold gating, K=4)
- Arm 3: Champion Learned Gate - Random Forest (RF Classifier, K=4)
- Arm 4: Champion Learned Gate - Logistic Regression (LR Classifier, K=4)

Scientific Verification Rules:
- Evaluates on Cases 90300 - 90799 (Zero overlap with any past training or testing).
- Models, weights, scalers, and thresholds locked prior to execution.
- Collision prioritized first (last_event == 'collision' -> COLLISION).
- Time limit prioritized (global_time >= 50.0s -> TIMEOUT).
- McNemar's exact two-sided statistical test computed for all candidate arms.
- Supports multi-worker parallel execution with incremental per-worker saving.
"""

import sys
import os
import copy
import math
import json
import time
import argparse
import subprocess
from pathlib import Path
import joblib
import numpy as np
import torch
from scipy.stats import binom_test

REPO_ROOT = Path(__file__).resolve().parent
KDARL_CLEAN = REPO_ROOT / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(REPO_ROOT)] + [p for p in sys.path if p not in (str(REPO_ROOT), str(KDARL_CLEAN), '')]

from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from run_selection_study_50cases import install_selection
from run_b0_counterfactual_study import (
    get_base_config, create_environment,
    compute_intent_compatibility, derive_macro_intent_from_action
)
from run_p3_cagis_benchmark import FEATURE_KEYS, heuristic_gate

RESULTS_DIR = REPO_ROOT / 'benchmark_results'
MODELS_DIR = REPO_ROOT / 'trained_models'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class Stage4ControllerPolicy(MambaRLPolicy):
    def __init__(self, mode='base', config=None, device='cpu'):
        super().__init__(config=config, device=device)
        self.mode = mode # 'base', 'heuristic', 'learned_rf', 'learned_lr'
        self.committed_intent = None
        self.dwell_steps_remaining = 0
        self.intervention_count = 0

        # State tracking buffers
        self.history_pos = []
        self.history_vel = []
        self.last_decision_features = None

        # Learned models
        self.rf_model = None
        self.lr_model = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.rf_threshold = 0.5
        self.lr_threshold = 0.5

    def init_learned_models(self):
        meta_file = MODELS_DIR / 'champion_metadata.json'
        with open(meta_file, 'r') as f:
            meta = json.load(f)

        self.scaler_mean = np.array(meta['scaler_mean'], dtype=np.float32)
        self.scaler_scale = np.array(meta['scaler_scale'], dtype=np.float32)

        if self.mode == 'learned_rf':
            self.rf_model = joblib.load(MODELS_DIR / 'champion_gating_rf.joblib')
            self.rf_threshold = float(meta['rf_model']['threshold'])
        elif self.mode == 'learned_lr':
            self.lr_model = joblib.load(MODELS_DIR / 'champion_gating_lr.joblib')
            self.lr_threshold = float(meta['lr_model']['threshold'])

    def reset_episode_stats(self):
        super().reset_episode_stats()
        self.committed_intent = None
        self.dwell_steps_remaining = 0
        self.intervention_count = 0
        self.history_pos.clear()
        self.history_vel.clear()
        self.last_decision_features = None

    def extract_observable_features(self, state: JointState, res_peek):
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
            closing_speed = -((nearest_h.px - rob.px)*rel_vx + (nearest_h.py - rob.py)*rel_vy) / max(0.01, math.hypot(nearest_h.px - rob.px, nearest_h.py - rob.py))
        else:
            dmin_curr = 999.0
            closing_speed = 0.0

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

        dist_to_goal = math.hypot(rob.gx - rob.px, rob.gy - rob.py)
        theta_g = math.atan2(rob.gy - rob.py, rob.gx - rob.px)
        theta_v = math.atan2(rob.vy, rob.vx) if math.hypot(rob.vx, rob.vy) > 0.05 else theta_g
        goal_heading_error = (theta_v - theta_g + math.pi) % (2*math.pi) - math.pi
        safe_actions_ratio = float((res_peek['clearance'] >= 0.20).float().mean().item())
        curr_speed = math.hypot(rob.vx, rob.vy)

        return {
            'dmin_curr': dmin_curr,
            'closing_speed': closing_speed,
            'travel_4s': t_4s,
            'disp_4s': d_4s,
            'ratio_4s': r_4s,
            'reversals_4s': rev_4s,
            'dist_to_goal': dist_to_goal,
            'goal_heading_error': goal_heading_error,
            'safe_actions_ratio': safe_actions_ratio,
            'current_speed': curr_speed
        }

    def score_sarl_candidates(self, state: JointState):
        base_res = super().score_sarl_candidates(state)
        rob = state.self_state

        if len(state.human_states):
            curr_phys_dists = [
                math.hypot(rob.px - h.px, rob.py - h.py) - rob.radius - h.radius
                for h in state.human_states
            ]
            current_physical_clearance = float(min(curr_phys_dists))
        else:
            current_physical_clearance = 999.0

        # Emergency clearance abort (<0.22m)
        if current_physical_clearance < 0.22:
            self.dwell_steps_remaining = 0
            self.committed_intent = None

        # Always record decision features
        features = self.extract_observable_features(state, base_res)
        self.last_decision_features = features

        # Intervention activation check
        if self.mode != 'base' and self.dwell_steps_remaining == 0:
            best_cmd = base_res['commands'][base_res['scores'].argmax().item()]
            cand_intent = derive_macro_intent_from_action(rob, best_cmd)

            should_intervene = False
            if self.mode == 'heuristic':
                should_intervene = heuristic_gate(features)
            elif self.mode == 'learned_rf':
                is_symptom = (features['ratio_4s'] >= 1.8 or features['reversals_4s'] >= 2)
                if is_symptom:
                    feat_vec = np.array([features[k] for k in FEATURE_KEYS], dtype=np.float32).reshape(1, -1)
                    feat_norm = (feat_vec - self.scaler_mean) / (self.scaler_scale + 1e-8)
                    prob = float(self.rf_model.predict_proba(feat_norm)[0, 1])
                    should_intervene = (prob >= self.rf_threshold)
            elif self.mode == 'learned_lr':
                is_symptom = (features['ratio_4s'] >= 1.8 or features['reversals_4s'] >= 2)
                if is_symptom:
                    feat_vec = np.array([features[k] for k in FEATURE_KEYS], dtype=np.float32).reshape(1, -1)
                    feat_norm = (feat_vec - self.scaler_mean) / (self.scaler_scale + 1e-8)
                    prob = float(self.lr_model.predict_proba(feat_norm)[0, 1])
                    should_intervene = (prob >= self.lr_threshold)

            if should_intervene:
                self.committed_intent = cand_intent
                self.dwell_steps_remaining = 4
                self.intervention_count += 1

        # Modulated intent commitment
        if self.dwell_steps_remaining > 0 and self.committed_intent is not None:
            compat = compute_intent_compatibility(base_res['commands'], rob, self.committed_intent).to(self.device)
            gate = max(0.0, min(1.0, (current_physical_clearance - 0.22) / 0.15))
            alpha = 0.10 * gate
            modulated_scores = base_res['scores'] + alpha * compat

            if self._phase in ('test', 'val', 'eval') and self.test_min_clearance > 0.0:
                safe_mask = base_res['clearance'] >= self.test_min_clearance
                if safe_mask.any():
                    modulated_scores = modulated_scores.masked_fill(~safe_mask, -1e9)

            base_res['scores'] = modulated_scores
            self.dwell_steps_remaining -= 1
        else:
            self.committed_intent = None
            self.dwell_steps_remaining = 0

        return base_res

    def predict(self, state):
        rob = state.self_state
        self.history_pos.append((rob.px, rob.py))
        act = super().predict(state)
        self.history_vel.append((act.vx, act.vy))
        return act

def load_stage4_policy(mode='base', device='cpu'):
    cfg = get_base_config()
    policy = Stage4ControllerPolicy(mode=mode, config=cfg, device=device)

    ckpt_path = REPO_ROOT / 'baseline_weights/gru-t24-seed443.pth'
    ckpt = torch.load(ckpt_path, map_location=device)
    raw_state = ckpt.get('policy_state', ckpt.get('model', ckpt))
    clean_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in raw_state.items()}
    policy.load_state_dict(clean_dict, strict=True)
    policy.eval()
    policy.build_action_space(1.0)
    policy.use_sarl_predict = True
    policy.set_phase('test')
    policy.test_min_clearance = 0.2
    policy.test_risk_lambda = 0.8
    policy.test_action_smoothing = 0.3
    install_selection(policy, 'ttc5')

    if mode in ('learned_rf', 'learned_lr'):
        policy.init_learned_models()

    return policy

def run_single_episode(policy, env, case_id):
    obs, _ = env.reset(options={'test_case': case_id})
    policy.reset_episode_stats()

    traj_x = [env.robot.px]
    traj_y = [env.robot.py]
    min_clearance = 999.0
    step = 0
    done = False
    last_info = None

    while not done and step < 201:
        step += 1
        state = JointState(env.robot.get_full_state(), [h.get_observable_state() for h in env.humans])
        if len(env.humans):
            dists = [math.hypot(env.robot.px - h.px, env.robot.py - h.py) - env.robot.radius - h.radius for h in env.humans]
            min_clearance = min(min_clearance, min(dists))

        act = policy.predict(state)
        traj_x.append(env.robot.px)
        traj_y.append(env.robot.py)

        obs, rew, term, trunc, last_info = env.step(act)
        done = term or trunc

    last_event = last_info.get('event', 'nothing') if last_info else 'nothing'
    if last_event == 'collision':
        outcome = 'COLLISION'
    elif last_event in ('reach_goal', 'reachgoal') or env.robot.reached_destination():
        outcome = 'SUCCESS'
    elif last_event in ('timeout', 'time_out') or env.global_time >= env.time_limit - 1e-4:
        outcome = 'TIMEOUT'
    else:
        outcome = 'TIMEOUT'

    final_time = float(env.global_time)
    p_arr = np.column_stack([traj_x[-40:], traj_y[-40:]]) if len(traj_x) >= 40 else np.column_stack([traj_x, traj_y])
    t_10s = float(np.sum(np.linalg.norm(np.diff(p_arr, axis=0), axis=1))) if len(p_arr) > 1 else 0.0
    d_10s = float(np.linalg.norm(p_arr[-1] - p_arr[0])) if len(p_arr) > 1 else 0.0
    is_stag = bool(outcome == 'TIMEOUT' and t_10s > 1.5 and d_10s < 0.35 and t_10s / max(d_10s, 0.01) > 4.0)

    return {
        'outcome': outcome,
        'final_time': round(final_time, 2),
        'total_steps': step,
        'min_clearance': round(float(min_clearance), 3),
        'stagnation': is_stag,
        'interventions': policy.intervention_count
    }

def run_chunk_benchmark(start_case, num_cases, device='cuda', out_file=None):
    env, env_config = create_environment()
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    policies = {
        'arm1_base': load_stage4_policy('base', device=device),
        'arm2_heuristic': load_stage4_policy('heuristic', device=device),
        'arm3_learned_rf': load_stage4_policy('learned_rf', device=device),
        'arm4_learned_lr': load_stage4_policy('learned_lr', device=device)
    }

    records = []
    t_start = time.time()

    for idx in range(num_cases):
        case_id = start_case + idx
        case_res = {'case_id': case_id}

        for arm_name, pol in policies.items():
            robot.set_policy(pol)
            robot.env = env
            env.set_robot(robot)
            ep_res = run_single_episode(pol, env, case_id)
            case_res[arm_name] = ep_res

        records.append(case_res)

        if (idx + 1) % 10 == 0 or idx == num_cases - 1:
            if out_file:
                with open(out_file, 'w') as f:
                    json.dump(records, f, indent=2)
                print(f"[{Path(out_file).name}] Completed {idx+1}/{num_cases} cases | Elapsed: {time.time()-t_start:.1f}s")

    return records

def mcnemar_exact_test(arm_a_outcomes, arm_b_outcomes):
    b = sum(1 for a, b in zip(arm_a_outcomes, arm_b_outcomes) if a == 'SUCCESS' and b != 'SUCCESS')
    c = sum(1 for a, b in zip(arm_a_outcomes, arm_b_outcomes) if a != 'SUCCESS' and b == 'SUCCESS')
    n = b + c
    if n == 0: return 1.0, b, c
    pval = binom_test(min(b, c), n, 0.5)
    return float(pval), b, c

def main():
    parser = argparse.ArgumentParser(description="Stage 4 Truly Independent 500-Case Evaluation")
    parser.add_argument('--start', type=int, default=90300)
    parser.add_argument('--num', type=int, default=500)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--chunk_mode', action='store_true')
    parser.add_argument('--out_file', type=str, default=None)
    args = parser.parse_args()

    if args.chunk_mode:
        run_chunk_benchmark(args.start, args.num, device=args.device, out_file=args.out_file)
        return

    total_cases = args.num
    num_workers = args.workers
    cases_per_worker = total_cases // num_workers

    print("=" * 105)
    print(f"STAGE 4: TRULY INDEPENDENT 500-CASE CLOSED-LOOP BENCHMARK (Cases {args.start} - {args.start + total_cases - 1})")
    print("=" * 105)
    print("Arm 1: Pure B0 Single-Stream Baseline (gru-t24-seed443.pth)")
    print("Arm 2: Best Heuristic Gating (r_4s >= 1.8, dmin >= 0.25, v_close <= 0.3, safe_actions >= 0.5)")
    print("Arm 3: Champion Learned Gating - Random Forest (Cost-Sensitive RF, K=4)")
    print("Arm 4: Champion Learned Gating - Logistic Regression (Cost-Sensitive LR, K=4)")
    print("-" * 105)

    chunks = []
    processes = []
    t_start = time.time()

    for w in range(num_workers):
        w_start = args.start + w * cases_per_worker
        w_out = RESULTS_DIR / f'stage4_part{w}.json'
        chunks.append({'worker_id': w, 'start': w_start, 'num': cases_per_worker, 'out_file': w_out})

        cmd = [
            sys.executable,
            str(REPO_ROOT / 'run_stage4_500cases_evaluation.py'),
            '--start', str(w_start),
            '--num', str(cases_per_worker),
            '--device', args.device,
            '--chunk_mode',
            '--out_file', str(w_out)
        ]
        log_f = open(RESULTS_DIR / f'stage4_worker_{w}.log', 'w')
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append((proc, log_f, w))

    print(f"Launched {num_workers} parallel workers. Monitoring execution...")

    for proc, log_f, wid in processes:
        proc.wait()
        log_f.close()
        print(f"Worker {wid} finished with returncode {proc.returncode}.")

    # Merge results
    merged_data = []
    for c in chunks:
        if c['out_file'].exists():
            with open(c['out_file'], 'r') as f:
                merged_data.extend(json.load(f))

    merged_data.sort(key=lambda x: x['case_id'])
    master_out = RESULTS_DIR / 'stage4_500cases_final_evaluation.json'
    with open(master_out, 'w') as f:
        json.dump(merged_data, f, indent=2)

    t_total = time.time() - t_start
    print("\n" + "=" * 105)
    print(f"STAGE 4 BENCHMARK COMPLETE: {len(merged_data)} CASES EVALUATED IN {t_total:.1f}s ({t_total/60:.2f} mins)")
    print(f"Saved master dataset to: {master_out}")
    print("=" * 105)

    # Comparative evaluation table
    arm_keys = ['arm1_base', 'arm2_heuristic', 'arm3_learned_rf', 'arm4_learned_lr']
    arm_labels = {
        'arm1_base': 'Pure B0 Baseline',
        'arm2_heuristic': 'Best Heuristic Gate',
        'arm3_learned_rf': 'Champion Learned RF Gate',
        'arm4_learned_lr': 'Champion Learned LR Gate'
    }

    print(f"{'Method / Arm':<28} | {'Success%':<9} | {'Collision%':<11} | {'Timeout%':<9} | {'Stagnation%':<12} | {'Time (s)':<9} | {'Interv/Ep'}")
    print("-" * 105)

    stats = {}
    for arm in arm_keys:
        outcomes = [d[arm]['outcome'] for d in merged_data]
        times = [d[arm]['final_time'] for d in merged_data if d[arm]['outcome'] == 'SUCCESS']
        sr = np.mean([1 if o == 'SUCCESS' else 0 for o in outcomes]) * 100
        cr = np.mean([1 if o == 'COLLISION' else 0 for o in outcomes]) * 100
        tr = np.mean([1 if o == 'TIMEOUT' else 0 for o in outcomes]) * 100
        sor = np.mean([1 if d[arm]['stagnation'] else 0 for d in merged_data]) * 100
        avg_t = np.mean(times) if times else 0.0
        avg_interv = np.mean([d[arm]['interventions'] for d in merged_data])

        stats[arm] = {'sr': sr, 'cr': cr, 'tr': tr, 'sor': sor, 'avg_time': avg_t, 'outcomes': outcomes}
        print(f"{arm_labels[arm]:<28} | {sr:7.1f}% | {cr:9.1f}% | {tr:7.1f}% | {sor:10.1f}% | {avg_t:7.2f}s  | {avg_interv:6.1f}")

    print("-" * 105)

    # McNemar's exact test
    print("\n" + "=" * 105)
    print("STATISTICAL SIGNIFICANCE TESTING (EXACT TWO-SIDED MCNEMAR TEST)")
    print("=" * 105)
    base_outs = stats['arm1_base']['outcomes']
    heu_outs = stats['arm2_heuristic']['outcomes']

    for arm in ['arm2_heuristic', 'arm3_learned_rf', 'arm4_learned_lr']:
        p_val, b, c = mcnemar_exact_test(base_outs, stats[arm]['outcomes'])
        print(f"{arm_labels[arm]:<28} vs Base B0: p = {p_val:.4f} (Base won {b}, Candidate won {c})")

    # Learned vs Heuristic head-to-head
    p_rf_h, b_rf, c_rf = mcnemar_exact_test(heu_outs, stats['arm3_learned_rf']['outcomes'])
    p_lr_h, b_lr, c_lr = mcnemar_exact_test(heu_outs, stats['arm4_learned_lr']['outcomes'])
    print("-" * 105)
    print(f"Learned RF vs Heuristic Gate: p = {p_rf_h:.4f} (Heuristic won {b_rf}, Learned RF won {c_rf})")
    print(f"Learned LR vs Heuristic Gate: p = {p_lr_h:.4f} (Heuristic won {b_lr}, Learned LR won {c_lr})")
    print("=" * 105)

if __name__ == '__main__':
    main()
