import os
import sys
import json
import math
import time
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path('/home/abc/workspace/autonomous_research_20261012')
sys.path.insert(0, str(REPO_ROOT))

from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from run_b0_counterfactual_study import create_environment
from run_stage4_500cases_evaluation import load_stage4_policy

REPLICATION_DIR = REPO_ROOT / 'replication'
OUT_RESULTS = REPLICATION_DIR / 'replay_gate_results.json'

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

def run_replay_gate(start_case=99000, num_cases=20, sim_rule='square_crossing', device='cpu'):
    assert_case_guard(start_case, num_cases)

    env, env_config = create_environment()
    env.test_sim = sim_rule
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    policy = load_stage4_policy('learned_lr', device=device)
    robot.set_policy(policy)
    robot.env = env
    env.set_robot(robot)

    case_reports = []
    all_passed = True
    first_failure = None

    print("=" * 95)
    print(f"REPLAY GATE CHECK: CASES {start_case}..{start_case + num_cases - 1} | SIM={sim_rule} | DEVICE={device}")
    print("=" * 95)

    for i in range(num_cases):
        case_id = start_case + i
        assert_case_guard(case_id, 1)

        # -------------------------------------------------------------
        # 1. Original Episode Run
        # -------------------------------------------------------------
        obs, _ = env.reset(options={'test_case': case_id})
        policy.reset_episode_stats()

        recorded_steps = []
        step = 0
        done = False

        while not done and step < 201:
            step += 1
            # Pre-step state
            pre_robot = (float(env.robot.px), float(env.robot.py), float(env.robot.vx), float(env.robot.vy), float(env.robot.theta), float(env.robot.radius))
            pre_humans = tuple(
                (float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.theta), float(h.radius))
                for h in env.humans
            )

            state = JointState(env.robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            act = policy.predict(state)
            act_tuple = (float(act.vx), float(act.vy))

            obs, rew, term, trunc, info = env.step(act)
            done = bool(term or trunc)
            event_str = str(info.get('event', 'nothing'))

            post_robot = (float(env.robot.px), float(env.robot.py), float(env.robot.vx), float(env.robot.vy))
            post_humans = tuple(
                (float(h.px), float(h.py), float(h.vx), float(h.vy))
                for h in env.humans
            )

            recorded_steps.append({
                'step': step,
                'pre_robot': pre_robot,
                'pre_humans': pre_humans,
                'act': act_tuple,
                'reward': float(rew),
                'term': bool(term),
                'trunc': bool(trunc),
                'event': event_str,
                'post_robot': post_robot,
                'post_humans': post_humans
            })

        final_event = info.get('event', 'nothing')

        # -------------------------------------------------------------
        # 2. Replay Run & Step-by-Step Verification
        # -------------------------------------------------------------
        obs, _ = env.reset(options={'test_case': case_id})
        policy.reset_episode_stats()

        max_diff_robot = 0.0
        max_diff_human = 0.0
        max_diff_act = 0.0
        max_diff_reward = 0.0
        discrete_term_match = True
        discrete_trunc_match = True
        discrete_event_match = True
        case_passed = True
        failure_reason = None

        replay_step = 0
        replay_done = False

        while not replay_done and replay_step < 201:
            replay_step += 1
            if replay_step > len(recorded_steps):
                case_passed = False
                failure_reason = f"Replay exceeded original step count ({replay_step} > {len(recorded_steps)})"
                break

            rec = recorded_steps[replay_step - 1]

            # Check pre-step robot state
            cur_pre_robot = (float(env.robot.px), float(env.robot.py), float(env.robot.vx), float(env.robot.vy), float(env.robot.theta), float(env.robot.radius))
            d_rob_pre = max(abs(a - b) for a, b in zip(cur_pre_robot, rec['pre_robot']))
            max_diff_robot = max(max_diff_robot, d_rob_pre)
            if d_rob_pre > 1e-9:
                case_passed = False
                failure_reason = f"Step {replay_step}: pre-step robot mismatch (diff={d_rob_pre:.2e} > 1e-9)"
                break

            # Check pre-step humans state
            cur_pre_humans = tuple(
                (float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.theta), float(h.radius))
                for h in env.humans
            )
            d_hum_pre = 0.0
            for h_cur, h_rec in zip(cur_pre_humans, rec['pre_humans']):
                d_hum_pre = max(d_hum_pre, max(abs(a - b) for a, b in zip(h_cur, h_rec)))
            max_diff_human = max(max_diff_human, d_hum_pre)
            if d_hum_pre > 1e-9:
                case_passed = False
                failure_reason = f"Step {replay_step}: pre-step human mismatch (diff={d_hum_pre:.2e} > 1e-9)"
                break

            # Policy predict
            state = JointState(env.robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            act = policy.predict(state)
            act_tuple = (float(act.vx), float(act.vy))
            d_act = max(abs(act_tuple[0] - rec['act'][0]), abs(act_tuple[1] - rec['act'][1]))
            max_diff_act = max(max_diff_act, d_act)
            if d_act > 1e-9:
                case_passed = False
                failure_reason = f"Step {replay_step}: policy action mismatch (diff={d_act:.2e} > 1e-9)"
                break

            # Step env
            obs, rew, term, trunc, info = env.step(act)
            replay_done = bool(term or trunc)
            cur_event = str(info.get('event', 'nothing'))

            # Check reward
            d_rew = abs(float(rew) - rec['reward'])
            max_diff_reward = max(max_diff_reward, d_rew)
            if d_rew > 1e-9:
                case_passed = False
                failure_reason = f"Step {replay_step}: reward mismatch (diff={d_rew:.2e} > 1e-9)"
                break

            # Check discrete flags
            if bool(term) != rec['term']:
                discrete_term_match = False
                case_passed = False
                failure_reason = f"Step {replay_step}: term flag mismatch ({bool(term)} != {rec['term']})"
                break
            if bool(trunc) != rec['trunc']:
                discrete_trunc_match = False
                case_passed = False
                failure_reason = f"Step {replay_step}: trunc flag mismatch ({bool(trunc)} != {rec['trunc']})"
                break
            if cur_event != rec['event']:
                discrete_event_match = False
                case_passed = False
                failure_reason = f"Step {replay_step}: event mismatch ('{cur_event}' != '{rec['event']}')"
                break

            # Check post-step robot state
            cur_post_robot = (float(env.robot.px), float(env.robot.py), float(env.robot.vx), float(env.robot.vy))
            d_rob_post = max(abs(a - b) for a, b in zip(cur_post_robot, rec['post_robot']))
            max_diff_robot = max(max_diff_robot, d_rob_post)
            if d_rob_post > 1e-9:
                case_passed = False
                failure_reason = f"Step {replay_step}: post-step robot mismatch (diff={d_rob_post:.2e} > 1e-9)"
                break

            # Check post-step humans state
            cur_post_humans = tuple(
                (float(h.px), float(h.py), float(h.vx), float(h.vy))
                for h in env.humans
            )
            d_hum_post = 0.0
            for h_cur, h_rec in zip(cur_post_humans, rec['post_humans']):
                d_hum_post = max(d_hum_post, max(abs(a - b) for a, b in zip(h_cur, h_rec)))
            max_diff_human = max(max_diff_human, d_hum_post)
            if d_hum_post > 1e-9:
                case_passed = False
                failure_reason = f"Step {replay_step}: post-step human mismatch (diff={d_hum_post:.2e} > 1e-9)"
                break

        if replay_step != len(recorded_steps) and case_passed:
            case_passed = False
            failure_reason = f"Total steps mismatch: replay={replay_step} != original={len(recorded_steps)}"

        rep = {
            'case_id': case_id,
            'total_steps': step,
            'event': final_event,
            'max_diff_robot_state': float(max_diff_robot),
            'max_diff_human_state': float(max_diff_human),
            'max_diff_action': float(max_diff_act),
            'max_diff_reward': float(max_diff_reward),
            'discrete_term_match': discrete_term_match,
            'discrete_trunc_match': discrete_trunc_match,
            'discrete_event_match': discrete_event_match,
            'passed': case_passed,
            'failure_reason': failure_reason
        }
        case_reports.append(rep)

        status_str = "PASS" if case_passed else f"FAIL ({failure_reason})"
        print(f"Case {case_id:5d} | Steps: {step:3d} | Outcome: {final_event:<12} | "
              f"MaxΔRob: {max_diff_robot:.1e} | MaxΔHum: {max_diff_human:.1e} | "
              f"MaxΔAct: {max_diff_act:.1e} | MaxΔRew: {max_diff_reward:.1e} | {status_str}")

        if not case_passed:
            all_passed = False
            if first_failure is None:
                first_failure = (case_id, failure_reason)
            break

    summary = {
        'gate_name': 'unintervened_replay_gate',
        'cases_tested': len(case_reports),
        'start_case': start_case,
        'end_case': start_case + len(case_reports) - 1,
        'sim_rule': sim_rule,
        'all_passed': all_passed,
        'first_failure': first_failure,
        'case_reports': case_reports
    }

    with open(OUT_RESULTS, 'w') as f:
        json.dump(summary, f, indent=2)

    print("=" * 95)
    print(f"REPLAY GATE RESULT: {'PASSED' if all_passed else 'FAILED'}")
    print(f"Saved results to: {OUT_RESULTS}")
    print("=" * 95)

    if not all_passed:
        print(f"ERROR: Replay gate check FAILED at case {first_failure[0]}: {first_failure[1]}")
        sys.exit(1)

    return summary

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_case', type=int, default=99000)
    parser.add_argument('--num_cases', type=int, default=20)
    parser.add_argument('--sim_rule', type=str, default='square_crossing')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    run_replay_gate(
        start_case=args.start_case,
        num_cases=args.num_cases,
        sim_rule=args.sim_rule,
        device=args.device
    )
