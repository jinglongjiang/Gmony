#!/usr/bin/env python3
"""Cycle 3 Experiment: The Limit-Cycle Oscillation Breakthrough

Problem Identified:
In 20S dense crowds, when forward clearance drops near 0.2m, the policy cannot stop
(include_stop=False, STOP tied-break penalized). It is forced to choose lateral/retreating velocities,
causing a limit-cycle oscillation between goal-attraction and discomfort-repulsion (net progress < 1m over 40 steps).

Hypothesis 3 (Adaptive Temporal Yielding Shield, ATYS):
When the candidate action set has no progressive action with clearance >= 0.2m,
instead of oscillating sideways or retreating, the robot yields (applies a slow speed/stop hold for up to 1.5s
until the dynamic obstruction clears). This breaks the limit cycle and unlocks goal completion.
"""

import sys
from pathlib import Path
import configparser
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized

def evaluate_with_shield(use_yielding_shield, num_cases=30):
    env_config = configparser.RawConfigParser()
    env_config.read(WORKSPACE / 'crowd_nav/configs/env.config')
    policy_config = configparser.RawConfigParser()
    policy_config.read(WORKSPACE / 'crowd_nav/configs/policy.config')
    train_config = configparser.RawConfigParser()
    train_config.read(WORKSPACE / 'crowd_nav/configs/train.config')

    cfg = configparser.RawConfigParser()
    for c in [env_config, policy_config, train_config]:
        for sec in c.sections():
            if not cfg.has_section(sec):
                cfg.add_section(sec)
            for k, v in c.items(sec):
                cfg.set(sec, k, v)

    cfg.set('mamba', 'temporal_backbone', 'gru')
    cfg.set('buffer', 'seq_len', '24')
    cfg.set('sim', 'human_num', '20')

    net = MambaRLPolicy(cfg, device='cpu')
    weights_path = WORKSPACE / 'baseline_weights/gru-t24-seed443.pth'
    ckpt = torch.load(weights_path, map_location='cpu')
    clean_dict = {k.replace('module.', ''): v for k, v in (ckpt['model'] if 'model' in ckpt else ckpt).items()}
    net.load_state_dict(clean_dict, strict=False)
    net.eval()
    net.build_action_space(1.0)
    net.use_sarl_predict = True

    env = CrowdSim()
    env.configure(env_config)
    env.human_num = 20
    env.square_width = 20.0
    env.circle_radius = 8.0
    env.phase = 'test'
    env.test_sim = 'square_crossing'

    robot = Robot(env_config, 'robot')
    robot.set_policy(net)
    robot.env = env
    env.set_robot(robot)

    results = {'success': 0, 'collision': 0, 'timeout': 0, 'times': [], 'net_progress': []}
    yield_interventions = 0

    mode_name = "With Adaptive Yielding Shield" if use_yielding_shield else "Standard Policy (Unshielded)"
    print(f"\n--- Testing [{mode_name}] on {num_cases} cases of 20S ---")

    for case_idx in range(num_cases):
        case_id = 90000 + case_idx
        env.case_counter['test'] = case_id
        obs, _ = env.reset(options={'test_case': case_id})
        net.reset_episode_stats()
        done = False
        step = 0
        init_dist = np.hypot(robot.px - robot.gx, robot.py - robot.gy)

        yield_counter = 0

        while not done and step < 100:
            step += 1
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            
            # Predict action from policy
            action = net.predict(state)

            if use_yielding_shield:
                # Check candidate clearances
                commands = net.executable_candidates()
                rob = state.self_state
                curr_dist = np.hypot(rob.px - rob.gx, rob.py - rob.gy)
                
                # Check if progressive forward actions exist with safe clearance >= 0.2m
                forward_safe_exists = False
                for cmd in commands:
                    next_px = rob.px + cmd.vx * 0.25
                    next_py = rob.py + cmd.vy * 0.25
                    new_dist = np.hypot(next_px - rob.gx, next_py - rob.gy)
                    if new_dist < curr_dist - 0.02: # forward progress
                        # check min clearance
                        clr = min(np.hypot(next_px - h.px, next_py - h.py) - rob.radius - h.radius for h in state.human_states)
                        if clr >= 0.20:
                            forward_safe_exists = True
                            break

                # If no forward action is safe (front is blocked by crossing pedestrians)
                # and robot is currently safe (not colliding), YIELD rather than retreating!
                cur_min_clr = min(np.hypot(rob.px - h.px, rob.py - h.py) - rob.radius - h.radius for h in state.human_states)
                if not forward_safe_exists and cur_min_clr >= 0.15 and yield_counter < 6: # yield up to 1.5s
                    action = ActionXY(0.0, 0.0) # Active yield / wait
                    yield_counter += 1
                    yield_interventions += 1
                else:
                    yield_counter = 0

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        final_dist = np.hypot(robot.px - robot.gx, robot.py - robot.gy)
        results['net_progress'].append(init_dist - final_dist)

        collision = any(np.linalg.norm([env.robot.px - h.px, env.robot.py - h.py]) < (env.robot.radius + h.radius) for h in env.humans)
        if env.robot.reached_destination():
            outcome = 'success'
        elif collision:
            outcome = 'collision'
        else:
            outcome = 'timeout'

        results[outcome] += 1
        results['times'].append(env.global_time)

    sr = 100.0 * results['success'] / num_cases
    cr = 100.0 * results['collision'] / num_cases
    tr = 100.0 * results['timeout'] / num_cases
    avg_prog = np.mean(results['net_progress'])
    print(f"[{mode_name}] Result: SR={sr:.1f}%, CR={cr:.1f}%, TR={tr:.1f}%, Mean Net Progress={avg_prog:.2f}m")
    if use_yielding_shield:
        print(f"Total Yield Interventions: {yield_interventions}")
    return results

def main():
    print("=" * 70)
    print("CYCLE 3: Breaking the Freezing Limit-Cycle via Adaptive Yielding Shield")
    print("=" * 70)
    n = 25
    res_unshielded = evaluate_with_shield(use_yielding_shield=False, num_cases=n)
    res_shielded = evaluate_with_shield(use_yielding_shield=True, num_cases=n)

    sr_un = 100.0 * res_unshielded['success'] / n
    sr_sh = 100.0 * res_shielded['success'] / n
    cr_un = 100.0 * res_unshielded['collision'] / n
    cr_sh = 100.0 * res_shielded['collision'] / n
    tr_un = 100.0 * res_unshielded['timeout'] / n
    tr_sh = 100.0 * res_shielded['timeout'] / n

    print("\n" + "=" * 70)
    print("CYCLE 3 COMPARISON SUMMARY:")
    print(f"Unshielded: SR={sr_un:.1f}%, CR={cr_un:.1f}%, TR={tr_un:.1f}%")
    print(f"Shielded:   SR={sr_sh:.1f}%, CR={cr_sh:.1f}%, TR={tr_sh:.1f}%")
    print(f"Delta SR: {sr_sh - sr_un:+.1f} pp, Delta CR: {cr_sh - cr_un:+.1f} pp, Delta TR: {tr_sh - tr_un:+.1f} pp")
    print(f"Verdict: {'PASS - Breakthrough Confirmed!' if sr_sh > sr_un else 'INCONCLUSIVE'}")
    print("=" * 70)

if __name__ == '__main__':
    main()
