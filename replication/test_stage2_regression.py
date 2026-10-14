import sys
from pathlib import Path
import math

REPO_ROOT = Path('/home/abc/workspace/autonomous_research_20261012')
sys.path.insert(0, str(REPO_ROOT))

from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from run_b0_counterfactual_study import create_environment
from run_stage4_500cases_evaluation import load_stage4_policy
from replication.stage2_policy import Stage2TerminationPolicy

def test_regression_parity():
    env, env_config = create_environment()
    env.test_sim = 'square_crossing'
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    # Policy 1: Original Stage 4 Learned LR
    orig_policy = load_stage4_policy('learned_lr', device='cpu')

    # Policy 2: Stage 2 Policy with forced_step4
    from replication.stage2_policy import load_stage2_policy
    s2_policy = load_stage2_policy(termination_mode='forced_step4', device='cpu')

    for case_id in range(99000, 99005):
        # Run original
        robot.set_policy(orig_policy)
        robot.env = env
        env.set_robot(robot)
        obs, _ = env.reset(options={'test_case': case_id})
        orig_policy.reset_episode_stats()

        orig_actions = []
        step = 0
        done = False
        while not done and step < 201:
            step += 1
            state = JointState(env.robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            act = orig_policy.predict(state)
            orig_actions.append((act.vx, act.vy))
            obs, rew, term, trunc, info = env.step(act)
            done = bool(term or trunc)
        orig_event = info.get('event')

        # Run Stage 2 forced_step4
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

        assert step == step_s2, f"Case {case_id}: step mismatch {step} vs {step_s2}"
        assert orig_event == s2_event, f"Case {case_id}: event mismatch {orig_event} vs {s2_event}"
        max_act_diff = max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(orig_actions, s2_actions))
        print(f"Case {case_id}: steps={step}, event={orig_event}, max_act_diff={max_act_diff:.1e}")
        assert max_act_diff < 1e-9, f"Case {case_id}: act diff {max_act_diff} > 1e-9"

    print("STAGE 2 REGRESSION GATE (FORCED STEP 4 == LR K=4): 100% PARITY PASSED!")

if __name__ == '__main__':
    test_regression_parity()
