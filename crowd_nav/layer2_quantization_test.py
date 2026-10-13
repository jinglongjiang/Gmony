#!/usr/bin/env python3
"""
debug.md Layer 2 Test: Action Quantization/Dequantization Verification

Based on findings:
- Layer 0: Continuous ORCA = 100% success
- Layer 2: Quantized ORCA = 0% success

This tests the ORCA → quantize → dequantize → environment pipeline.
"""
import sys
import os
import numpy as np
sys.path.insert(0, '/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')
os.chdir('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav')

print("=== DEBUG.MD LAYER 2: QUANTIZATION TEST ===")

try:
    import configparser
    from crowd_sim.envs.crowd_sim import CrowdSim
    from crowd_sim.envs.utils.robot import Robot
    from crowd_sim.envs.utils.state import JointState
    from crowd_nav.policy.policy_factory import policy_factory
    from crowd_nav.contracts import discrete_index_to_action, GRID

    # Load config
    cfg = configparser.RawConfigParser()
    cfg.read(['configs/env.config', 'configs/policy.config', 'configs/train.config'])

    # Build environment
    env = CrowdSim()
    env.configure(cfg)
    robot = Robot(cfg, 'robot')
    env.set_robot(robot)

    # Create ORCA
    orca = policy_factory['orca']()
    orca.configure(cfg)
    orca.seed(42)

    print("\n=== Step 1: Test Continuous ORCA ===")
    obs = env.reset(phase='test')
    state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])

    # Get continuous ORCA action
    continuous_action = orca.predict(state)
    print(f"Continuous ORCA action: vx={continuous_action.vx:.3f}, vy={continuous_action.vy:.3f}")

    # Test continuous action in environment
    obs_next, reward, terminated, truncated, info = env.step(continuous_action)
    print(f"Continuous result: reward={reward:.3f}, terminated={terminated}, event={info.get('event', 'none')}")

    print("\n=== Step 2: Test Action Quantization ===")
    # Reset environment for fair comparison
    obs = env.reset(phase='test')
    state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])

    # Get the same continuous action
    continuous_action = orca.predict(state)
    print(f"Original continuous: vx={continuous_action.vx:.3f}, vy={continuous_action.vy:.3f}")

    # Find closest discrete action
    best_idx = 0
    best_distance = float('inf')

    print("\n=== Checking action grid ===")
    print(f"GRID config: {GRID}")

    for idx in range(80):  # 80 discrete actions
        discrete_vx, discrete_vy = discrete_index_to_action(idx, **GRID)
        distance = (discrete_vx - continuous_action.vx)**2 + (discrete_vy - continuous_action.vy)**2

        if distance < best_distance:
            best_distance = distance
            best_idx = idx

        if idx < 10:  # Show first 10 for debugging
            print(f"  Action {idx}: ({discrete_vx:.3f}, {discrete_vy:.3f}), distance={distance:.4f}")

    # Get the best quantized action
    quantized_vx, quantized_vy = discrete_index_to_action(best_idx, **GRID)
    print(f"\nBest quantized action: idx={best_idx}, vx={quantized_vx:.3f}, vy={quantized_vy:.3f}")
    print(f"Quantization error: {np.sqrt(best_distance):.4f}")

    # Create proper ActionXY object for environment
    from crowd_sim.envs.utils.action import ActionXY
    quantized_action = ActionXY(quantized_vx, quantized_vy)

    # Test quantized action in environment
    obs_next, reward, terminated, truncated, info = env.step(quantized_action)
    print(f"Quantized result: reward={reward:.3f}, terminated={terminated}, event={info.get('event', 'none')}")

    print("\n=== Step 3: Compare Results ===")
    print("If continuous ORCA works but quantized fails, the issue is in:")
    print("1. Action grid configuration (v_min, v_max, speeds)")
    print("2. Quantization resolution (80 actions insufficient)")
    print("3. Action space mismatch between ORCA and environment")

    print("\n=== Step 4: Test Multiple Episodes ===")
    continuous_successes = 0
    quantized_successes = 0
    episodes = 10

    for ep in range(episodes):
        # Test continuous
        obs = env.reset(phase='test')
        terminated = False
        steps = 0
        max_steps = 100

        while not terminated and steps < max_steps:
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            action = orca.predict(state)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        if terminated and "reach" in str(info.get("event", "")).lower():
            continuous_successes += 1

        # Test quantized
        obs = env.reset(phase='test')
        terminated = False
        steps = 0

        while not terminated and steps < max_steps:
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            action = orca.predict(state)

            # Quantize action
            best_idx = 0
            best_distance = float('inf')
            for idx in range(80):
                discrete_vx, discrete_vy = discrete_index_to_action(idx, **GRID)
                distance = (discrete_vx - action.vx)**2 + (discrete_vy - action.vy)**2
                if distance < best_distance:
                    best_distance = distance
                    best_idx = idx

            quantized_vx, quantized_vy = discrete_index_to_action(best_idx, **GRID)
            quantized_action = ActionXY(quantized_vx, quantized_vy)

            obs, reward, terminated, truncated, info = env.step(quantized_action)
            steps += 1

        if terminated and "reach" in str(info.get("event", "")).lower():
            quantized_successes += 1

    print(f"\n=== LAYER 2 RESULTS ===")
    print(f"Continuous ORCA: {continuous_successes}/{episodes} = {continuous_successes/episodes:.3f}")
    print(f"Quantized ORCA:  {quantized_successes}/{episodes} = {quantized_successes/episodes:.3f}")

    if continuous_successes > quantized_successes:
        print("❌ LAYER 2 PROBLEM CONFIRMED: Quantization breaks ORCA performance")
        print("debug.md diagnosis correct: Issue is in action quantization, not environment")
    else:
        print("✅ Quantization OK, problem elsewhere")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()