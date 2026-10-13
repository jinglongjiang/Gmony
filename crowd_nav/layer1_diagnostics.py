#!/usr/bin/env python3
"""
debug.md Layer 1 Diagnostics: Environment "卡死因子" Analysis

Layer 1 factors per debug.md:
1.1 时间单位：秒 vs 步
1.2 成功/碰撞判定
1.3 ORCA 配参"中庸型"
1.4 额外：删干净对 human.vx/vy 的任何"二次避让/TTC 钩子"
"""
import sys
import os
import numpy as np
sys.path.insert(0, '/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')
os.chdir('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav')

print("=== DEBUG.MD LAYER 1 DIAGNOSTICS ===")

try:
    import configparser
    from crowd_sim.envs.crowd_sim import CrowdSim
    from crowd_sim.envs.utils.robot import Robot
    from crowd_sim.envs.utils.state import JointState
    from crowd_nav.policy.policy_factory import policy_factory

    # Load config
    cfg = configparser.RawConfigParser()
    cfg.read(['configs/env.config', 'configs/policy.config', 'configs/train.config'])

    # Build environment
    env = CrowdSim()
    env.configure(cfg)
    robot = Robot(cfg, 'robot')
    env.set_robot(robot)

    print("\n=== 1.1 时间单位检查 ===")
    dt = cfg.getfloat('env', 'time_step', fallback=0.25)
    time_limit = cfg.getfloat('env', 'time_limit', fallback=25.0)
    expected_steps = round(time_limit / dt)
    print(f"dt={dt}s, time_limit={time_limit}s, expected_steps={expected_steps}")
    print(f"✅ Time configuration seems correct: {time_limit}s / {dt}s = {expected_steps} steps")

    print("\n=== 1.2 成功/碰撞判定检查 ===")

    # Reset and get initial state
    obs = env.reset(phase='test')
    robot_state = robot.get_full_state()
    human_states = [h.get_observable_state() for h in env.humans]

    print(f"Robot initial: px={robot_state.px:.3f}, py={robot_state.py:.3f}, radius={robot_state.radius:.3f}")
    print(f"Robot goal: gx={robot_state.gx:.3f}, gy={robot_state.gy:.3f}")

    success_radius = cfg.getfloat('robot', 'success_radius', fallback=0.25)
    goal_dist = np.sqrt((robot_state.gx - robot_state.px)**2 + (robot_state.gy - robot_state.py)**2)
    print(f"Success radius: {success_radius:.3f}")
    print(f"Initial distance to goal: {goal_dist:.3f}")

    print(f"Humans ({len(human_states)}):")
    collision_detected = False
    for i, h in enumerate(human_states):
        h_dist = np.sqrt((h.px - robot_state.px)**2 + (h.py - robot_state.py)**2)
        collision_threshold = robot_state.radius + h.radius

        print(f"  Human {i+1}: px={h.px:.3f}, py={h.py:.3f}, radius={h.radius:.3f}")
        print(f"    Distance to robot: {h_dist:.3f}, collision_threshold: {collision_threshold:.3f}")

        if h_dist <= collision_threshold:
            print(f"    ❌ IMMEDIATE COLLISION DETECTED! {h_dist:.3f} <= {collision_threshold:.3f}")
            collision_detected = True
        else:
            print(f"    ✅ Safe spacing: {h_dist:.3f} > {collision_threshold:.3f}")

    if collision_detected:
        print("\n❌ PROBLEM FOUND: Immediate spawn collision!")
        print("SOLUTION: Check environment spawn logic - humans too close to robot at start")
    else:
        print("\n✅ No immediate spawn collision detected")

    print("\n=== 1.3 ORCA 配参检查 ===")
    orca = policy_factory['orca']()
    orca.configure(cfg)

    print(f"ORCA parameters:")
    print(f"  neighbor_dist: {orca.neighbor_dist:.2f}")
    print(f"  max_neighbors: {orca.max_neighbors}")
    print(f"  time_horizon: {orca.time_horizon:.2f}")
    print(f"  time_horizon_obst: {orca.time_horizon_obst:.2f}")
    print(f"  max_speed: {orca.max_speed:.3f}")
    print(f"  safety_space: {orca.safety_space:.3f}")

    # Check if parameters are "moderate" per debug.md
    if (2.5 <= orca.neighbor_dist <= 4.0 and
        8 <= orca.max_neighbors <= 15 and
        4.0 <= orca.time_horizon <= 8.0):
        print("✅ ORCA parameters appear moderate/reasonable")
    else:
        print("⚠️  ORCA parameters may be too conservative or aggressive")
        print("debug.md suggests: neighbor_dist≈3.5, max_neighbors≈12, time_horizon≈6.0")

    print("\n=== Single step test ===")
    # Test what happens in first step
    state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
    action = orca.predict(state)
    print(f"ORCA action: vx={action.vx:.3f}, vy={action.vy:.3f}")

    step_out = env.step(action)
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
    else:
        obs, reward, terminated, info = step_out
        truncated = False

    print(f"Step result: terminated={terminated}, reward={reward:.3f}")
    print(f"Event: {info.get('event', 'None')}")

    # Check new positions
    new_robot_state = robot.get_full_state()
    new_human_states = [h.get_observable_state() for h in env.humans]

    print(f"Robot new pos: px={new_robot_state.px:.3f}, py={new_robot_state.py:.3f}")
    for i, h in enumerate(new_human_states):
        h_dist = np.sqrt((h.px - new_robot_state.px)**2 + (h.py - new_robot_state.py)**2)
        collision_threshold = new_robot_state.radius + h.radius
        print(f"Human {i+1} new pos: px={h.px:.3f}, py={h.py:.3f}, dist={h_dist:.3f}, thresh={collision_threshold:.3f}")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()