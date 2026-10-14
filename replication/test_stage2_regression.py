import sys
import json
import math
import time
from pathlib import Path

REPO_ROOT = Path('/home/abc/workspace/autonomous_research_20261012')
sys.path.insert(0, str(REPO_ROOT))

from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from run_b0_counterfactual_study import create_environment
from run_stage4_500cases_evaluation import load_stage4_policy
from replication.stage2_policy import load_stage2_policy

OUT_JSON = REPO_ROOT / 'replication' / 'stage2_regression_gate.json'

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

def test_regression_parity(start_case=99000, num_cases=20):
    assert_case_guard(start_case, num_cases)
    env, env_config = create_environment()
    env.test_sim = 'square_crossing'
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    # Policy 1: Original Stage 4 Learned LR
    orig_policy = load_stage4_policy('learned_lr', device='cpu')

    # Policy 2: Stage 2 Policy with forced_step4
    s2_policy = load_stage2_policy(termination_mode='forced_step4', device='cpu')

    results = []
    all_passed = True
    overall_max_diff = 0.0
    t0 = time.time()

    for case_id in range(start_case, start_case + num_cases):
        # 1. Run original LR gated
        robot.set_policy(orig_policy)
        robot.env = env
        env.set_robot(robot)
        obs, _ = env.reset(options={'test_case': case_id})
        orig_policy.reset_episode_stats()

        orig_actions = []
        step_orig = 0
        done_orig = False
        while not done_orig and step_orig < 201:
            step_orig += 1
            state = JointState(env.robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            act = orig_policy.predict(state)
            orig_actions.append((act.vx, act.vy))
            obs, rew, term, trunc, info = env.step(act)
            done_orig = bool(term or trunc)
        orig_event = info.get('event')

        # 2. Run Stage 2 forced_step4
        robot.set_policy(s2_policy)
        robot.env = env
        env.set_robot(robot)
        obs, _ = env.reset(options={'test_case': case_id})
        s2_policy.reset_episode_stats()

        s2_actions = []
        step_s2 = 0
        done_s2 = False
        while not done_s2 and step_s2 < 201:
            step_s2 += 1
            state = JointState(env.robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            act = s2_policy.predict(state)
            s2_actions.append((act.vx, act.vy))
            obs, rew, term, trunc, info = env.step(act)
            done_s2 = bool(term or trunc)
        s2_event = info.get('event')

        steps_match = (step_orig == step_s2)
        event_match = (orig_event == s2_event)
        max_act_diff = max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(orig_actions, s2_actions)) if steps_match else float('inf')
        case_passed = (steps_match and event_match and max_act_diff < 1e-9)

        if not case_passed:
            all_passed = False

        overall_max_diff = max(overall_max_diff, max_act_diff)

        print(f"Case {case_id}: steps={step_orig}/{step_s2}, event={orig_event}/{s2_event}, max_diff={max_act_diff:.1e}, passed={case_passed}")

        results.append({
            'case_id': case_id,
            'steps_orig': step_orig,
            'steps_s2': step_s2,
            'event_orig': orig_event,
            'event_s2': s2_event,
            'max_act_diff': float(max_act_diff),
            'passed': bool(case_passed)
        })

    summary = {
        'gate': 'Stage 2 Regression Gate (Forced Step 4 == Learned LR K=4)',
        'status': 'PASSED' if all_passed else 'FAILED',
        'cases_tested': num_cases,
        'case_range': [start_case, start_case + num_cases - 1],
        'overall_max_action_diff': float(overall_max_diff),
        'tolerance': 1e-9,
        'all_passed': bool(all_passed),
        'elapsed_seconds': round(time.time() - t0, 2),
        'results': results
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"STAGE 2 REGRESSION GATE: {'100% PARITY PASSED' if all_passed else 'FAILED'} (max diff: {overall_max_diff:.2e})")
    print(f"Results written to: {OUT_JSON}")
    print("=" * 80)

    if not all_passed:
        sys.exit(1)

if __name__ == '__main__':
    test_regression_parity(99000, 20)
