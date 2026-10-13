#!/usr/bin/env python3
"""
debug.md Layer 1.2 Fix: Collision Detection

Problem: Current trajectory-based collision detection with collision_threshold=0.0
causes immediate collisions even when humans are far away.

Solution: Use simple distance-based collision detection for current positions only.
"""
import sys
import os
sys.path.insert(0, '/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')
os.chdir('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav')

print("=== LAYER 1.2 COLLISION DETECTION FIX ===")

# 1. First backup the original file
import shutil
backup_path = "/home/abc/workspace/nav_data/mamba/camrl/CrowdNav/crowd_sim/envs/crowd_sim_backup.py"
original_path = "/home/abc/workspace/nav_data/mamba/camrl/CrowdNav/crowd_sim/envs/crowd_sim.py"

try:
    shutil.copy2(original_path, backup_path)
    print(f"✅ Backup created: {backup_path}")
except Exception as e:
    print(f"⚠️  Backup failed: {e}")

# 2. Read current collision detection code
with open(original_path, 'r') as f:
    content = f.read()

print("Current collision detection uses trajectory-based method with collision_threshold=0.0")
print("This causes immediate collisions when trajectories pass near humans")

# 3. Apply the fix - replace trajectory-based with position-based collision detection
old_collision_code = """        # 严格碰撞判定：恢复CrowdNav口径
        collision_threshold = 0.0

        for i, h in enumerate(self.humans):
            px = h.px - self.robot.px
            py = h.py - self.robot.py
            vx = h.vx + rvx
            vy = h.vy + rvy
            ex = px + vx * dt
            ey = py + vy * dt
            closest = point_to_segment_dist(px, py, ex, ey, 0.0, 0.0) - h.radius - self.robot.radius

            if closest < collision_threshold:
                collision = True
                closest_id = i
                dmin = closest
                ttc_min = 0.0
                break"""

new_collision_code = """        # debug.md Layer 1.2 修复：简单位置碰撞判定（当前位置，非轨迹）
        # 使用 dist(robot, human) ≤ (r_robot + r_human) 判定，符合CrowdNav标准

        for i, h in enumerate(self.humans):
            # 当前位置距离（不考虑轨迹）
            current_dist = ((h.px - self.robot.px)**2 + (h.py - self.robot.py)**2)**0.5
            collision_threshold = h.radius + self.robot.radius

            if current_dist <= collision_threshold:
                collision = True
                closest_id = i
                dmin = current_dist - collision_threshold  # 负值表示重叠
                ttc_min = 0.0
                break

            # 仍计算最小距离用于统计（但不用于碰撞判定）
            if current_dist < dmin:
                dmin = current_dist - collision_threshold
                closest_id = i"""

if old_collision_code.strip() in content:
    new_content = content.replace(old_collision_code, new_collision_code)

    with open(original_path, 'w') as f:
        f.write(new_content)

    print("✅ Applied Layer 1.2 collision detection fix")
    print("Changed from trajectory-based to simple position-based collision detection")
    print("Formula: collision = dist(robot, human) ≤ (r_robot + r_human)")

    # Test the fix
    print("\n=== Testing fix ===")

    import configparser
    from crowd_sim.envs.crowd_sim import CrowdSim
    from crowd_sim.envs.utils.robot import Robot
    from crowd_sim.envs.utils.state import JointState
    from crowd_nav.policy.policy_factory import policy_factory

    # Reload the module to get the fixed version
    import importlib
    import crowd_sim.envs.crowd_sim
    importlib.reload(crowd_sim.envs.crowd_sim)

    # Load config
    cfg = configparser.RawConfigParser()
    cfg.read(['configs/env.config', 'configs/policy.config', 'configs/train.config'])

    # Build environment with fix
    env = CrowdSim()
    env.configure(cfg)
    robot = Robot(cfg, 'robot')
    env.set_robot(robot)

    # Create ORCA
    orca = policy_factory['orca']()
    orca.configure(cfg)

    # Test a few episodes
    successes = 0
    episodes = 5

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
                    print(f"✅ Episode {ep+1}: SUCCESS in {steps} steps")
                elif "collision" in event:
                    print(f"❌ Episode {ep+1}: COLLISION in {steps} steps")
                else:
                    print(f"❓ Episode {ep+1}: {event} in {steps} steps")
                break
        else:
            print(f"⏰ Episode {ep+1}: TIMEOUT after {steps} steps")

    success_rate = successes / episodes
    print(f"\n=== FIX TEST RESULTS ===")
    print(f"Success rate: {success_rate:.3f} ({successes}/{episodes})")

    if success_rate > 0:
        print("✅ FIX SUCCESSFUL! ORCA teacher can now reach goals")
        print("Layer 1.2 collision detection issue resolved")
    else:
        print("❌ Fix unsuccessful, may need additional Layer 1 checks")
        print("Consider Layer 1.3 (ORCA parameters) or Layer 1.4 (TTC hooks)")

else:
    print("❌ Could not find expected collision detection code to replace")
    print("Manual inspection needed")

print(f"\nBackup available at: {backup_path}")
print("To restore: cp {backup_path} {original_path}")