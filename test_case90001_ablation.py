#!/usr/bin/env python3
"""Targeted Forensics on Case 90001 across B0, B1, B2, B3"""

import sys
from pathlib import Path
import configparser
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parent
KDARL_CLEAN = WORKSPACE / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(WORKSPACE)] + [p for p in sys.path if p not in (str(WORKSPACE), str(KDARL_CLEAN), '')]

from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from run_selection_study_50cases import install_selection
from strat_mamba_dual_stream import StratMambaDualStreamPolicy
from learnable_intent_mamba import LICSStratMambaPolicy, HeuristicHysteresisPolicy

def run_single_case(policy, case_id=90001, policy_name="Policy"):
    env_config = configparser.RawConfigParser()
    env_config.read(KDARL_CLEAN / 'crowd_nav/configs/env.config')
    env = CrowdSim()
    env.configure(env_config)
    env.time_limit = 50.0
    env.human_num = 20
    env.square_width = 10.0
    env.circle_radius = 4.0
    env.phase = 'test'
    env.test_sim = 'square_crossing'

    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    robot.env = env
    env.set_robot(robot)

    obs, _ = env.reset(options={'test_case': case_id})
    policy.reset_episode_stats()

    traj_x = []
    traj_y = []
    actions = []
    done = False
    step = 0

    init_dist = math_hypot = np.hypot(robot.px - robot.gx, robot.py - robot.gy)

    for step in range(1, 202):
        state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
        action = policy.predict(state)
        traj_x.append(robot.px)
        traj_y.append(robot.py)
        actions.append((action.vx, action.vy))

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    final_dist = np.hypot(robot.px - robot.gx, robot.py - robot.gy)
    collision = any(np.linalg.norm([env.robot.px - h.px, env.robot.py - h.py]) < (env.robot.radius + h.radius) for h in env.humans)
    if env.robot.reached_destination():
        outcome = 'SUCCESS'
    elif collision:
        outcome = 'COLLISION'
    else:
        outcome = 'TIMEOUT'

    # Compute trajectory statistics in the last 40 steps (10 seconds)
    last_pts = np.column_stack([traj_x[-40:], traj_y[-40:]]) if len(traj_x) >= 40 else np.column_stack([traj_x, traj_y])
    total_len = np.sum(np.linalg.norm(np.diff(last_pts, axis=0), axis=1)) if len(last_pts) > 1 else 0.0
    net_disp = np.linalg.norm(last_pts[-1] - last_pts[0]) if len(last_pts) > 1 else 0.0
    ratio = total_len / max(net_disp, 0.01)

    print(f"[{policy_name}] Case {case_id}: Outcome = {outcome}, Steps = {step}, Time = {env.global_time:.2f}s")
    print(f"    Init Dist = {init_dist:.2f}m, Final Dist = {final_dist:.2f}m")
    print(f"    Last 10s Travel Distance = {total_len:.3f}m, Net Displacement = {net_disp:.3f}m, Ratio = {ratio:.1f}")
    return outcome, step, env.global_time

def main():
    env_config = configparser.RawConfigParser()
    env_config.read(KDARL_CLEAN / 'crowd_nav/configs/env.config')
    policy_config = configparser.RawConfigParser()
    policy_config.read(KDARL_CLEAN / 'crowd_nav/configs/policy.config')
    train_config = configparser.RawConfigParser()
    train_config.read(KDARL_CLEAN / 'crowd_nav/configs/train.config')

    cfg = configparser.RawConfigParser()
    for c in [env_config, policy_config, train_config]:
        for sec in c.sections():
            if not cfg.has_section(sec): cfg.add_section(sec)
            for k, v in c.items(sec): cfg.set(sec, k, v)

    cfg.set('env', 'time_limit', '50')
    cfg.set('mamba', 'temporal_backbone', 'gru')
    cfg.set('buffer', 'seq_len', '24')
    cfg.set('sim', 'human_num', '20')

    weights_path = WORKSPACE / 'baseline_weights/gru-t24-seed443.pth'

    # 1. Arm B0: Base Single Stream TTC-5
    b0 = MambaRLPolicy(cfg, device='cpu')
    ckpt = torch.load(weights_path, map_location='cpu')
    raw_state = ckpt.get('policy_state', ckpt.get('model', ckpt))
    clean_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in raw_state.items()}
    b0.load_state_dict(clean_dict, strict=True)
    b0.eval()
    b0.build_action_space(1.0)
    b0.use_sarl_predict = True
    b0.set_phase('test')
    b0.test_min_clearance = 0.2
    b0.test_risk_lambda = 0.8
    b0.test_action_smoothing = 0.3
    install_selection(b0, 'ttc5')

    # 2. Arm B1: StratMamba Dual Stream
    b1 = StratMambaDualStreamPolicy(cfg, device='cpu', backbone='gru')
    b1.load_pretrained_weights(weights_path)
    b1.eval()
    b1.build_action_space(1.0)
    b1.use_sarl_predict = True
    b1.set_phase('test')
    b1.test_min_clearance = 0.2
    b1.test_risk_lambda = 0.8
    b1.test_action_smoothing = 0.3
    install_selection(b1, 'ttc5')

    # 3. Arm B2: LICS
    b2 = LICSStratMambaPolicy(cfg, device='cpu', backbone='gru', hysteresis_margin=0.15)
    b2.load_pretrained_weights(weights_path)
    b2.eval()
    b2.build_action_space(1.0)
    b2.use_sarl_predict = True
    b2.set_phase('test')
    b2.test_min_clearance = 0.2
    b2.test_risk_lambda = 0.8
    b2.test_action_smoothing = 0.3
    install_selection(b2, 'ttc5')

    # 4. Arm B3: Heuristic Hysteresis (dwell K=4)
    b3 = HeuristicHysteresisPolicy(cfg, device='cpu', backbone='gru', dwell_k=4)
    b3.load_pretrained_weights(weights_path)
    b3.eval()
    b3.build_action_space(1.0)
    b3.use_sarl_predict = True
    b3.set_phase('test')
    b3.test_min_clearance = 0.2
    b3.test_risk_lambda = 0.8
    b3.test_action_smoothing = 0.3
    install_selection(b3, 'ttc5')

    print("=================================================================")
    print("COMPARATIVE EVALUATION ON TEST CASE 90001 (KNOWN OSCILLATION CASE)")
    print("=================================================================")
    run_single_case(b0, 90001, "Arm B0 (Base TTC-5)")
    run_single_case(b1, 90001, "Arm B1 (Dual-Stream StratMamba W_fuse=0)")
    run_single_case(b2, 90001, "Arm B2 (LICS Untrained Head)")
    run_single_case(b3, 90001, "Arm B3 (Heuristic Hysteresis K=4)")
    print("=================================================================")

if __name__ == '__main__':
    main()
