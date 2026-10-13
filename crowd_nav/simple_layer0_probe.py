#!/usr/bin/env python3
"""
Simplified Layer 0 Probe: Direct ORCA teacher test
"""
import sys
import os
sys.path.insert(0, '/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')
os.chdir('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav')

print("=== DEBUG.MD LAYER 0 PROBE ===")
print("Testing continuous ORCA teacher reachability")

try:
    import configparser
    from crowd_sim.envs.crowd_sim import CrowdSim
    from crowd_sim.envs.utils.robot import Robot
    from crowd_sim.envs.utils.state import JointState
    from crowd_nav.policy.policy_factory import policy_factory

    print("✅ All imports successful")

    # Load config
    cfg = configparser.RawConfigParser()
    cfg.read(['configs/env.config', 'configs/policy.config', 'configs/train.config'])
    print("✅ Config loaded")

    # Build environment
    env = CrowdSim()
    env.configure(cfg)
    robot = Robot(cfg, 'robot')
    env.set_robot(robot)
    print("✅ Environment created")

    # Create ORCA
    orca = policy_factory['orca']()
    orca.configure(cfg)
    orca.seed(42)
    print("✅ ORCA policy created")

    # Test single episode
    successes = 0
    episodes = 10  # Small test first

    for ep in range(episodes):
        obs = env.reset(phase='test')
        terminated = False
        steps = 0
        max_steps = 100

        while not terminated and steps < max_steps:
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            action = orca.predict(state)

            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
            else:
                obs, reward, terminated, info = step_out
                truncated = False

            steps += 1

            if terminated:
                event = str(info.get("event", "")).lower()
                if "success" in event or "reach" in event:
                    successes += 1
                    print(f"✅ Episode {ep+1}: SUCCESS in {steps} steps, event={event}")
                else:
                    print(f"❌ Episode {ep+1}: FAILED in {steps} steps, event={event}")
                break
        else:
            print(f"⏰ Episode {ep+1}: TIMEOUT after {steps} steps")

    success_rate = successes / episodes
    print(f"\n=== RESULTS ===")
    print(f"Episodes: {episodes}")
    print(f"Successes: {successes}")
    print(f"Success rate: {success_rate:.3f}")

    if success_rate >= 0.6:
        print("✅ SUCCESS_RATE ≥ 0.6 → Teacher can reach goal → Problem in Layer 2 (quantization)")
    else:
        print("❌ SUCCESS_RATE < 0.6 → Teacher cannot reach goal → Problem in Layer 1 (environment)")
        print("Layer 1 checks needed:")
        print("- Time units (dt=0.25s, timeout logic)")
        print("- Success/collision detection")
        print("- ORCA parameters too conservative")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()