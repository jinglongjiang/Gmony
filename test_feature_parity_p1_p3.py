#!/usr/bin/env python3
"""Stage 1: Automated Feature Parity Test (P1 Offline Extractor vs P3 Online Controller)

Verifies that from the exact same simulation state, the P1 reference feature extractor
and the P3 BenchmarkB0Policy online feature extractor produce 100% identical 10-dimensional
features on EVERY simulation step, including during active intent dwell.

Audits:
- Feature 1: dmin_curr
- Feature 2: closing_speed
- Feature 3: travel_4s
- Feature 4: disp_4s
- Feature 5: ratio_4s
- Feature 6: reversals_4s
- Feature 7: dist_to_goal
- Feature 8: goal_heading_error
- Feature 9: safe_actions_ratio
- Feature 10: current_speed
"""

import sys
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
from run_p3_cagis_benchmark import load_benchmark_policy, create_environment, FEATURE_KEYS

RESULTS_DIR = REPO_ROOT / 'p0_audit_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def compute_p1_reference_features(robot, env, history_pos, history_vel, res_peek):
    """Exact reference implementation from run_b0_counterfactual_study.py."""
    rob = robot
    if len(env.humans):
        phys_clearances = [
            math.hypot(rob.px - h.px, rob.py - h.py) - rob.radius - h.radius
            for h in env.humans
        ]
        dmin_curr = float(min(phys_clearances))
        nearest_h = env.humans[int(np.argmin(phys_clearances))]
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

    dist_to_goal = math.hypot(rob.gx - rob.px, rob.gy - rob.py)
    theta_g = math.atan2(rob.gy - rob.py, rob.gx - rob.px)
    theta_v = math.atan2(rob.vy, rob.vx) if math.hypot(rob.vx, rob.vy) > 0.05 else theta_g
    goal_heading_error = (theta_v - theta_g + math.pi) % (2*math.pi) - math.pi
    safe_actions_ratio = float((res_peek['clearance'] >= 0.20).float().mean().item())
    current_speed = math.hypot(rob.vx, rob.vy)

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
        'current_speed': current_speed
    }

def run_parity_test_for_case(case_id, mode='heuristic', device='cpu'):
    env, env_config = create_environment()
    policy = load_benchmark_policy(mode=mode, device=device)
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    robot.env = env
    env.set_robot(robot)

    obs, _ = env.reset(options={'test_case': case_id})
    policy.reset_episode_stats()

    # Ground truth external reference history
    ref_history_pos = []
    ref_history_vel = []

    step = 0
    done = False
    max_errors = {k: 0.0 for k in FEATURE_KEYS}
    total_checks = 0

    while not done and step < 80:
        step += 1
        px, py = robot.px, robot.py
        ref_history_pos.append((px, py))

        state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])

        # Execute decision step
        act = policy.predict(state)

        # Reference velocity buffer prior to this decision
        p1_feat = compute_p1_reference_features(robot, env, ref_history_pos, ref_history_vel, {'clearance': policy._last_clearance if hasattr(policy, '_last_clearance') else torch.tensor([0.3])})
        ref_history_vel.append((act.vx, act.vy))

        # Check internal buffer length alignment
        assert len(policy.history_pos) == len(ref_history_pos), f"Step {step}: history_pos length mismatch ({len(policy.history_pos)} vs {len(ref_history_pos)})"
        assert len(policy.history_vel) == len(ref_history_vel), f"Step {step}: history_vel length mismatch ({len(policy.history_vel)} vs {len(ref_history_vel)})"

        # Policy decision features recorded during predict()
        p3_feat = policy.last_decision_features
        assert p3_feat is not None, f"Step {step}: policy.last_decision_features is None!"

        # Verify exact parity across all 10 observable features
        total_checks += 1
        for k in FEATURE_KEYS:
            diff = abs(p1_feat[k] - p3_feat[k])
            max_errors[k] = max(max_errors[k], diff)
            assert diff < 1e-5, f"Step {step} Feature {k} mismatch! P1={p1_feat[k]} vs P3={p3_feat[k]} (diff={diff})"

        obs, rew, term, trunc, _ = env.step(act)
        done = term or trunc

    return {
        'case_id': case_id,
        'mode': mode,
        'steps_tested': step,
        'total_checks': total_checks,
        'max_errors': {k: float(v) for k, v in max_errors.items()},
        'passed': True
    }

def main():
    print("=" * 85)
    print("STAGE 1: FEATURE PARITY VERIFICATION (P1 REFERENCE vs P3 ONLINE CONTROLLER)")
    print("=" * 85)

    test_cases = [90000, 90001, 90005, 90010]
    modes = ['heuristic', 'fixed', 'cagis']
    all_results = []
    t_start = time.time()

    for cid in test_cases:
        for m in modes:
            res = run_parity_test_for_case(cid, mode=m, device='cpu')
            all_results.append(res)
            max_err = max(res['max_errors'].values())
            print(f"Case {cid} | Mode: {m:<10} | Steps: {res['steps_tested']:<3} | Max Diff across all 10 features: {max_err:.8f} | ✓ PASS")

    out_file = RESULTS_DIR / 'feature_parity_verification.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 85)
    print(f"100% FEATURE PARITY VERIFIED ACROSS ALL {len(all_results)} CONDITIONS IN {time.time()-t_start:.2f}s!")
    print(f"Saved audit log to: {out_file}")
    print("=" * 85)

if __name__ == '__main__':
    main()
