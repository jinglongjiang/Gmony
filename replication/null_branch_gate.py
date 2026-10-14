import os
import sys
import json
import math
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import torch

REPO_ROOT = Path('/home/abc/workspace/autonomous_research_20261012')
sys.path.insert(0, str(REPO_ROOT))

from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import JointState
from run_b0_counterfactual_study import create_environment
from replication.stage2_policy import load_stage2_policy

REPLICATION_DIR = REPO_ROOT / 'replication'
OUT_GATE_JSON = REPLICATION_DIR / 'null_branch_gate.json'

FORBIDDEN_RANGES = [
    (92000, 92499),
    (90000, 90299),
    (93000, 93199),
    (95000, 96499)
]

def assert_case_guard(start_case, num_cases):
    end_case = start_case + num_cases - 1
    for f_start, f_end in FORBIDDEN_RANGES:
        if not (end_case < f_start or start_case > f_end):
            raise ValueError(
                f"SAFETY VIOLATION: Case range [{start_case}, {end_case}] overlaps "
                f"with strictly prohibited range [{f_start}, {f_end}]!"
            )

def run_case_null_branch_gate(case_id, gamma=0.99):
    assert_case_guard(case_id, 1)

    env, env_config = create_environment()
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    policy = load_stage2_policy(termination_mode='fixed_k4')
    robot.set_policy(policy)

    # 1. Run full reference trajectory (standard LR gated K=4)
    obs, _ = env.reset(options={'test_case': case_id})
    policy.reset_episode_stats()

    orig_records = []
    step = 0
    done = False

    while not done and step < 201:
        state = JointState(env.robot.get_full_state(), [h.get_observable_state() for h in env.humans])
        rob = state.self_state
        human_states = [[h.px, h.py, h.vx, h.vy, h.radius] for h in state.human_states]
        rob_state = [rob.px, rob.py, rob.vx, rob.vy, rob.radius]

        # Capture complete internal policy state BEFORE prediction
        saved_pol_state = policy.capture_internal_state()

        was_committed = (policy.committed_intent is not None)
        dwell_held = policy.dwell_steps_held
        committed_intent = policy.committed_intent

        # Step is eligible for termination sampling if commitment is currently active
        is_sampled = (was_committed and dwell_held >= 1)

        act = policy.predict(state)
        obs, rew, term, trunc, info = env.step(act)
        done = bool(term or trunc)

        orig_records.append({
            'step': step,
            'state_rob': rob_state,
            'state_humans': human_states,
            'saved_pol_state': saved_pol_state,
            'was_committed': was_committed,
            'dwell_held': dwell_held,
            'committed_intent': committed_intent,
            'is_sampled': is_sampled,
            'action': ActionXY(act.vx, act.vy),
            'reward': float(rew),
            'done': done,
            'event': info.get('event', 'nothing')
        })
        step += 1

    total_steps = len(orig_records)
    returns_continue = [0.0] * total_steps
    cur_g = 0.0
    for t in reversed(range(total_steps)):
        cur_g = orig_records[t]['reward'] + gamma * cur_g
        returns_continue[t] = cur_g

    # 2. For every sampled step, test the null branch replay
    sampled_indices = [t for t, r in enumerate(orig_records) if r['is_sampled']]
    step_details = []
    case_passed = True
    case_max_g_diff = 0.0
    case_max_act_diff = 0.0
    case_max_pos_diff = 0.0
    case_max_rew_diff = 0.0

    for target_t in sampled_indices:
        target_rec = orig_records[target_t]
        # Replay prefix 0..target_t - 1
        obs, _ = env.reset(options={'test_case': case_id})
        policy.reset_episode_stats()

        for p in range(target_t):
            env.step(orig_records[p]['action'])

        # Restore policy state using exact method
        policy.restore_internal_state(target_rec['saved_pol_state'])

        # Rollout null branch from target_t to episode termination
        cur_step = target_t
        branch_rewards = []
        branch_done = False
        step_max_act_diff = 0.0
        step_max_pos_diff = 0.0
        step_max_rew_diff = 0.0
        events_match = True

        while not branch_done and cur_step < 201:
            st = JointState(env.robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            orig_step_rec = orig_records[cur_step]

            # Verify environment state match before predict
            rob_diff = math.hypot(
                st.self_state.px - orig_step_rec['state_rob'][0],
                st.self_state.py - orig_step_rec['state_rob'][1]
            )
            step_max_pos_diff = max(step_max_pos_diff, rob_diff)

            b_act = policy.predict(st)
            obs, b_rew, term, trunc, info = env.step(b_act)
            branch_done = bool(term or trunc)
            branch_rewards.append(float(b_rew))

            # Action difference
            adiff = abs(b_act.vx - orig_step_rec['action'].vx) + abs(b_act.vy - orig_step_rec['action'].vy)
            step_max_act_diff = max(step_max_act_diff, adiff)

            # Reward difference
            rdiff = abs(float(b_rew) - orig_step_rec['reward'])
            step_max_rew_diff = max(step_max_rew_diff, rdiff)

            # Event match
            if info.get('event') != orig_step_rec['event']:
                events_match = False

            cur_step += 1

        # Compute discounted return of the null branch
        g_null = 0.0
        for r in reversed(branch_rewards):
            g_null = r + gamma * g_null

        g_diff = abs(g_null - returns_continue[target_t])
        step_passed = (
            g_diff < 1e-9 and
            step_max_act_diff < 1e-9 and
            step_max_pos_diff < 1e-9 and
            step_max_rew_diff < 1e-9 and
            events_match and
            cur_step == total_steps
        )

        if not step_passed:
            case_passed = False

        case_max_g_diff = max(case_max_g_diff, g_diff)
        case_max_act_diff = max(case_max_act_diff, step_max_act_diff)
        case_max_pos_diff = max(case_max_pos_diff, step_max_pos_diff)
        case_max_rew_diff = max(case_max_rew_diff, step_max_rew_diff)

        step_details.append({
            'step': target_t,
            'dwell_held': target_rec['dwell_held'],
            'intent': int(target_rec['committed_intent']) if target_rec['committed_intent'] is not None else None,
            'g_continue': float(returns_continue[target_t]),
            'g_null': float(g_null),
            'g_diff': float(g_diff),
            'max_act_diff': float(step_max_act_diff),
            'max_pos_diff': float(step_max_pos_diff),
            'max_rew_diff': float(step_max_rew_diff),
            'events_match': events_match,
            'event_orig': target_rec['event'],
            'event_null': orig_records[-1]['event'],
            'passed': bool(step_passed)
        })

    return {
        'case_id': case_id,
        'total_steps': total_steps,
        'sampled_steps': len(sampled_indices),
        'case_passed': bool(case_passed),
        'max_g_diff': float(case_max_g_diff),
        'max_act_diff': float(case_max_act_diff),
        'max_pos_diff': float(case_max_pos_diff),
        'max_rew_diff': float(case_max_rew_diff),
        'step_details': step_details
    }

def worker_fn(case_id):
    try:
        res = run_case_null_branch_gate(case_id)
        print(f"[NullBranchGate] Case {case_id}: {res['sampled_steps']} sampled steps, "
              f"max_act_diff={res['max_act_diff']:.2e}, max_g_diff={res['max_g_diff']:.2e}, "
              f"passed={res['case_passed']}", flush=True)
        return res
    except Exception as e:
        print(f"[NullBranchGate] ERROR in case {case_id}: {e}", flush=True)
        raise

def main():
    cases = list(range(99000, 99020))
    assert_case_guard(cases[0], len(cases))
    print(f"=== Starting Null Branch Gate for {len(cases)} cases ({cases[0]} - {cases[-1]}) | WORKERS=2 ===", flush=True)
    t0 = time.time()

    # Constraint: parallel workers = 2
    num_workers = 2
    case_results = {}

    with Pool(processes=num_workers) as pool:
        for res in pool.imap_unordered(worker_fn, cases):
            case_results[res['case_id']] = res

            # Incremental save
            current_list = list(case_results.values())
            total_sampled = sum(r['sampled_steps'] for r in current_list)
            passed_sampled = sum(sum(1 for s in r['step_details'] if s['passed']) for r in current_list)
            all_passed = all(r['case_passed'] for r in current_list)
            overall_max_g_diff = max(r['max_g_diff'] for r in current_list) if current_list else 0.0
            overall_max_act_diff = max(r['max_act_diff'] for r in current_list) if current_list else 0.0
            overall_max_pos_diff = max(r['max_pos_diff'] for r in current_list) if current_list else 0.0
            overall_max_rew_diff = max(r['max_rew_diff'] for r in current_list) if current_list else 0.0

            summary = {
                'gate_name': 'Stage 2 Null Branch Replay Gate',
                'status': 'PASSED' if (len(case_results) == len(cases) and all_passed) else ('IN_PROGRESS' if len(case_results) < len(cases) else 'FAILED'),
                'total_cases': len(cases),
                'completed_cases': len(case_results),
                'case_range': [cases[0], cases[-1]],
                'total_sampled_steps': total_sampled,
                'passed_sampled_steps': passed_sampled,
                'all_passed': bool(all_passed),
                'tolerance': 1e-9,
                'max_return_diff': float(overall_max_g_diff),
                'max_action_diff': float(overall_max_act_diff),
                'max_robot_pos_diff': float(overall_max_pos_diff),
                'max_reward_diff': float(overall_max_rew_diff),
                'elapsed_seconds': round(time.time() - t0, 2)
            }

            gate_payload = {
                'summary': summary,
                'cases': case_results
            }

            with open(OUT_GATE_JSON, 'w') as f:
                json.dump(gate_payload, f, indent=2)

    all_passed = all(r['case_passed'] for r in case_results.values()) and (len(case_results) == len(cases))
    print(f"\n=== Null Branch Gate Evaluation Complete (All 20 Cases Finished) ===", flush=True)
    print(f"Status: {summary['status']}", flush=True)
    print(f"Sampled Steps: {passed_sampled} / {total_sampled} passed", flush=True)
    print(f"Max G diff: {overall_max_g_diff:.2e}", flush=True)
    print(f"Max Act diff: {overall_max_act_diff:.2e}", flush=True)
    print(f"Elapsed: {summary['elapsed_seconds']}s", flush=True)
    print(f"Saved results to: {OUT_GATE_JSON}", flush=True)

    if not all_passed:
        sys.exit(1)

if __name__ == '__main__':
    main()
