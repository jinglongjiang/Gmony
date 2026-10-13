#!/usr/bin/env python3
"""Phase P3: Closed-Loop Benchmarking of CAGIS vs Baselines on Fresh Unseen Cases

Rigorous 4-Arm Comparative Benchmark on 100 Fresh Unseen Cases (Cases 90200 - 90299):
- Arm 1: Base B0 (Single-Stream GRU, Zero Intervention)
- Arm 2: B0 + Fixed Dwell-4 (Unconditional 4-step intent commitment)
- Arm 3: B0 + Heuristic Gated (Hand-crafted threshold gating, K=4)
- Arm 4: B0 + CAGIS (Counterfactual Advantage-Guided Intent Switching with Meta-MLP, K=4)

Strict Experimental Integrity:
- All arms evaluated on identical random seeds / test case IDs.
- Collision prioritized first: last_event == 'collision' -> COLLISION.
- Timeout priority: global_time >= 50.0s -> TIMEOUT.
- Audits McNemar's exact test, collision surge suppression, and stagnation recovery.
- Supports 4 parallel worker processes for maximum throughput.
"""

import sys
import copy
import math
import json
import time
import argparse
import subprocess
from pathlib import Path
import configparser
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent
KDARL_CLEAN = REPO_ROOT / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(REPO_ROOT)] + [p for p in sys.path if p not in (str(REPO_ROOT), str(KDARL_CLEAN), '')]

from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from run_selection_study_50cases import install_selection
from run_b0_counterfactual_study import (
    get_base_config, create_environment,
    compute_intent_compatibility, derive_macro_intent_from_action,
    INTENT_BYPASS_LEFT, INTENT_DIRECT, INTENT_BYPASS_RIGHT, INTENT_YIELD, INTENT_NAMES
)
from analyze_feature_predictability_p2 import CAGISMetaMLP, FEATURE_KEYS, heuristic_gate

RESULTS_DIR = REPO_ROOT / 'benchmark_results'
MODELS_DIR = REPO_ROOT / 'trained_models'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class BenchmarkB0Policy(MambaRLPolicy):
    """B0 Controller with configurable intervention mode: None, Fixed, Heuristic, or CAGIS."""
    def __init__(self, mode='base', config=None, device='cpu'):
        super().__init__(config=config, device=device)
        self.mode = mode  # 'base', 'fixed', 'heuristic', 'cagis'
        self.committed_intent = None
        self.dwell_steps_remaining = 0
        self.intervention_count = 0
        self.meta_model = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.cagis_threshold = 0.5
        
        # State tracking buffers
        self.history_pos = []
        self.history_vel = []

    def init_meta_model(self, meta_checkpoint, device='cpu'):
        ckpt = torch.load(meta_checkpoint, map_location=device)
        self.meta_model = CAGISMetaMLP(in_features=10).to(device)
        self.meta_model.load_state_dict(ckpt['model_state_dict'])
        self.meta_model.eval()
        self.scaler_mean = np.array(ckpt['scaler_mean'], dtype=np.float32)
        self.scaler_scale = np.array(ckpt['scaler_scale'], dtype=np.float32)
        self.cagis_threshold = float(ckpt.get('threshold', 0.5))

    def reset_episode_stats(self):
        super().reset_episode_stats()
        self.committed_intent = None
        self.dwell_steps_remaining = 0
        self.intervention_count = 0
        self.history_pos.clear()
        self.history_vel.clear()

    def extract_observable_features(self, state: JointState, res_peek):
        rob = state.self_state
        px, py = rob.px, rob.py
        self.history_pos.append((px, py))
        
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

        # Safety abort: if clearance tight (<0.22m), cancel intent immediately
        if current_physical_clearance < 0.22:
            self.dwell_steps_remaining = 0
            self.committed_intent = None

        # Check for intervention activation if not currently dwelling
        if self.mode != 'base' and self.dwell_steps_remaining == 0:
            features = self.extract_observable_features(state, base_res)
            best_cmd = base_res['commands'][base_res['scores'].argmax().item()]
            cand_intent = derive_macro_intent_from_action(rob, best_cmd)

            should_intervene = False
            if self.mode == 'fixed':
                should_intervene = True
            elif self.mode == 'heuristic':
                should_intervene = heuristic_gate(features)
            elif self.mode == 'cagis' and self.meta_model is not None:
                feat_vec = np.array([features[k] for k in FEATURE_KEYS], dtype=np.float32)
                feat_norm = (feat_vec - self.scaler_mean) / (self.scaler_scale + 1e-8)
                with torch.no_grad():
                    inp = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
                    prob = torch.sigmoid(self.meta_model(inp)).item()
                should_intervene = (prob >= self.cagis_threshold)

            if should_intervene:
                self.committed_intent = cand_intent
                self.dwell_steps_remaining = 4
                self.intervention_count += 1

        # Apply intent modulation if active
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
        act = super().predict(state)
        self.history_vel.append((act.vx, act.vy))
        return act

def load_benchmark_policy(mode='base', device='cpu'):
    cfg = get_base_config()
    policy = BenchmarkB0Policy(mode=mode, config=cfg, device=device)
    
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

    if mode == 'cagis':
        ckpt_meta = MODELS_DIR / 'cagis_meta_policy.pth'
        policy.init_meta_model(ckpt_meta, device=device)

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

    # Preload all 4 policies once
    policies = {
        'arm1_base': load_benchmark_policy('base', device=device),
        'arm2_fixed': load_benchmark_policy('fixed', device=device),
        'arm3_heuristic': load_benchmark_policy('heuristic', device=device),
        'arm4_cagis': load_benchmark_policy('cagis', device=device)
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

        if (idx + 1) % 5 == 0 or idx == num_cases - 1:
            if out_file:
                with open(out_file, 'w') as f:
                    json.dump(records, f, indent=2)
                print(f"[{Path(out_file).name}] Completed {idx+1}/{num_cases} cases | Elapsed: {time.time()-t_start:.1f}s")

    return records

def mcnemar_exact_test(arm_a_outcomes, arm_b_outcomes):
    """Computes exact two-sided McNemar p-value for binary success parity."""
    # b: A success, B fail; c: A fail, B success
    b = sum(1 for a, b in zip(arm_a_outcomes, arm_b_outcomes) if a == 'SUCCESS' and b != 'SUCCESS')
    c = sum(1 for a, b in zip(arm_a_outcomes, arm_b_outcomes) if a != 'SUCCESS' and b == 'SUCCESS')
    n = b + c
    if n == 0:
        return 1.0, b, c
    # Exact binomial two-sided test
    from scipy.stats import binomtest
    res = binomtest(min(b, c), n, 0.5, alternative='two-sided')
    return float(res.pvalue), b, c

def main():
    parser = argparse.ArgumentParser(description="Phase P3 CAGIS 4-Arm Benchmark")
    parser.add_argument('--start', type=int, default=90200)
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--chunk_mode', action='store_true', help="Internal worker flag")
    parser.add_argument('--out_file', type=str, default=None)
    args = parser.parse_args()

    if args.chunk_mode:
        run_chunk_benchmark(args.start, args.num, device=args.device, out_file=args.out_file)
        return

    # Master runner mode
    total_cases = args.num
    num_workers = args.workers
    cases_per_worker = total_cases // num_workers

    print("=" * 95)
    print(f"PHASE P3: 4-ARM CAGIS BENCHMARK ON {total_cases} FRESH UNSEEN CASES (Cases {args.start} - {args.start + total_cases - 1})")
    print("=" * 95)
    print("Arm 1: Pure B0 Single-Stream Baseline")
    print("Arm 2: B0 + Fixed Dwell-4 (Unconditional Intent Commitment)")
    print("Arm 3: B0 + Heuristic Gated Intent Commitment")
    print("Arm 4: B0 + CAGIS (Learned Meta-MLP Guided Intent Commitment)")
    print("-" * 95)

    chunks = []
    processes = []
    t_start = time.time()

    for w in range(num_workers):
        w_start = args.start + w * cases_per_worker
        w_out = RESULTS_DIR / f'p3_cagis_part{w}.json'
        chunks.append({'worker_id': w, 'start': w_start, 'num': cases_per_worker, 'out_file': w_out})

        cmd = [
            sys.executable,
            str(REPO_ROOT / 'run_p3_cagis_benchmark.py'),
            '--start', str(w_start),
            '--num', str(cases_per_worker),
            '--device', args.device,
            '--chunk_mode',
            '--out_file', str(w_out)
        ]
        log_f = open(RESULTS_DIR / f'p3_worker_{w}.log', 'w')
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append((proc, log_f, w))

    for proc, log_f, wid in processes:
        proc.wait()
        log_f.close()
        print(f"P3 Worker {wid} finished with code {proc.returncode}.")

    # Merge results
    merged_data = []
    for c in chunks:
        if c['out_file'].exists():
            with open(c['out_file'], 'r') as f:
                merged_data.extend(json.load(f))

    merged_data.sort(key=lambda x: x['case_id'])
    master_out = RESULTS_DIR / 'p3_cagis_benchmark_results.json'
    with open(master_out, 'w') as f:
        json.dump(merged_data, f, indent=2)

    t_total = time.time() - t_start
    print("\n" + "=" * 95)
    print(f"PHASE P3 BENCHMARK COMPLETE: {len(merged_data)} CASES EVALUATED IN {t_total:.1f}s ({t_total/60:.2f} mins)")
    print(f"Saved results to: {master_out}")
    print("=" * 95)

    # Compute comparative metrics
    arm_keys = ['arm1_base', 'arm2_fixed', 'arm3_heuristic', 'arm4_cagis']
    arm_labels = {
        'arm1_base': 'Pure B0 Baseline',
        'arm2_fixed': 'B0 + Fixed Dwell-4',
        'arm3_heuristic': 'B0 + Heuristic Gate',
        'arm4_cagis': 'B0 + CAGIS (Learned)'
    }

    N = len(merged_data)
    print(f"{'Method / Arm':<25} | {'Success%':<9} | {'Collision%':<11} | {'Timeout%':<9} | {'Stagnation%':<12} | {'Time (s)':<9} | {'Interv/Ep'}")
    print("-" * 95)

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
        print(f"{arm_labels[arm]:<25} | {sr:7.1f}% | {cr:9.1f}% | {tr:7.1f}% | {sor:10.1f}% | {avg_t:7.2f}s  | {avg_interv:6.1f}")

    print("-" * 95)

    # Statistical significance testing
    print("\n" + "=" * 95)
    print("STATISTICAL SIGNIFICANCE (EXACT TWO-SIDED MCNEMAR TEST)")
    print("=" * 95)
    
    base_outs = stats['arm1_base']['outcomes']
    for arm in ['arm2_fixed', 'arm3_heuristic', 'arm4_cagis']:
        p_val, b, c = mcnemar_exact_test(base_outs, stats[arm]['outcomes'])
        print(f"{arm_labels[arm]:<25} vs Base B0: p = {p_val:.4f} (Base won {b}, Candidate won {c})")

    # CAGIS vs Heuristic
    p_c_vs_h, b_ch, c_ch = mcnemar_exact_test(stats['arm3_heuristic']['outcomes'], stats['arm4_cagis']['outcomes'])
    print(f"{arm_labels['arm4_cagis']:<25} vs Heuristic: p = {p_c_vs_h:.4f} (Heuristic won {b_ch}, CAGIS won {c_ch})")
    print("=" * 95)

if __name__ == '__main__':
    main()
