#!/usr/bin/env python3
"""
场景自适应调参版本 - 每个场景专属优化
"""
import logging, argparse, configparser, os, re, torch, numpy as np, random, time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState

def compute_ttc(robot, human):
    rx, ry, rvx, rvy = robot.px, robot.py, robot.vx, robot.vy
    hx, hy, hvx, hvy = human.px, human.py, human.vx, human.vy
    rel_x, rel_y = hx - rx, hy - ry
    rel_vx, rel_vy = hvx - rvx, hvy - rvy
    dist = np.hypot(rel_x, rel_y) + 1e-6
    closing = -(rel_x*rel_vx + rel_y*rel_vy) / dist
    return dist / (closing + 1e-6) if closing > 0.1 else 1e9

def test_episode(env, robot, policy, test_case_idx):
    if hasattr(policy, 'reset_episode_stats'):
        policy.reset_episode_stats()
    reset_result = env.reset(seed=test_case_idx)
    ob = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    done, steps = False, 0
    max_steps = 500

    while not done and steps < max_steps:
        robot_state = robot.get_full_state()
        humans = env.humans
        # TTC风险排序
        if len(humans) > 5:
            sorted_humans = sorted(humans, key=lambda h: (compute_ttc(robot, h), np.hypot(h.px - robot.px, h.py - robot.py)))[:5]
        else:
            dists = [np.hypot(h.px - robot.px, h.py - robot.py) for h in humans]
            sorted_humans = [h for _, h in sorted(zip(dists, humans), key=lambda pair: pair[0])]
        human_states = [h.get_observable_state() for h in sorted_humans]
        state = JointState(robot_state, human_states)
        action = policy.predict(state)

        step_result = env.step(action)
        if len(step_result) == 5:
            ob, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            ob, reward, done, info = step_result
        steps += 1

    outcome = 'timeout'
    if isinstance(info, dict):
        event = info.get('event', '').lower()
        if any(x in event for x in ['reach_goal', 'success', 'reachgoal']):
            outcome = 'success'
        elif any(x in event for x in ['collision']):
            outcome = 'collision'
    else:
        if info.__class__.__name__ == 'ReachGoal':
            outcome = 'success'
        elif info.__class__.__name__ == 'Collision':
            outcome = 'collision'
    return outcome

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='mamba')
    parser.add_argument('--model_dir', type=str, default='runs/mamba_vl')
    parser.add_argument('--weights', type=str, default='latest')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--test_case', type=int, default=None)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    print("[场景自适应调参] 每个场景专属优化")
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    # 场景专属配置
    test_cases = [
        {'desc': 'baseline_circle', 'sim': 'circle_crossing', 'human_num': 5,  'circle_radius': 4.0, 'v_pref': 1.0, 'time_limit': 25},
        {'desc': 'baseline_square', 'sim': 'square_crossing', 'human_num': 10, 'square_width': 10.0, 'v_pref': 0.6, 'time_limit': 50},
        {'desc': 'dense_circle',    'sim': 'circle_crossing', 'human_num': 10, 'circle_radius': 4.0, 'v_pref': 0.9, 'time_limit': 35},
        {'desc': 'dense_square',    'sim': 'square_crossing', 'human_num': 20, 'square_width': 10.0, 'v_pref': 0.5, 'time_limit': 60},
        {'desc': 'large_circle',    'sim': 'circle_crossing', 'human_num': 12, 'circle_radius': 6.0, 'v_pref': 0.9, 'time_limit': 40},
        {'desc': 'large_square',    'sim': 'square_crossing', 'human_num': 20, 'square_width': 15.0, 'v_pref': 0.5, 'time_limit': 60},
    ]

    if args.test_case is not None and 0 <= args.test_case < len(test_cases):
        test_cases = [test_cases[args.test_case]]

    base_seed = int(time.time() * 1000) % (2**31)
    random.seed(base_seed)
    np.random.seed(base_seed)

    for case_idx, case in enumerate(test_cases):
        print(f"\n{'='*70}")
        print(f"场景 [{case_idx}]: {case['desc']} | {case['human_num']}人 | v_pref={case['v_pref']} | T={case['time_limit']}s")
        print(f"{'='*70}")

        env_config = configparser.RawConfigParser()
        env_config.read(args.env_config)
        policy_config = configparser.RawConfigParser()
        policy_config.read(args.policy_config)

        if not policy_config.has_section('buffer'): policy_config.add_section('buffer')
        policy_config.set('buffer', 'seq_len', '12')
        if not policy_config.has_section('robot'): policy_config.add_section('robot')
        policy_config.set('robot', 'v_pref', str(case['v_pref']))
        if not env_config.has_section('env'): env_config.add_section('env')
        env_config.set('env', 'time_step', '0.25')
        env_config.set('env', 'time_limit', str(case['time_limit']))
        if not env_config.has_section('robot'): env_config.add_section('robot')
        env_config.set('robot', 'v_pref', str(case['v_pref']))

        for section in policy_config.sections():
            if not env_config.has_section(section): env_config.add_section(section)
            for key, value in policy_config.items(section): env_config.set(section, key, value)

        if not env_config.has_section('sim'): env_config.add_section('sim')
        env_config.set('sim', 'test_sim', case['sim'])
        env_config.set('sim', 'human_num', str(case['human_num']))
        if 'circle_radius' in case: env_config.set('sim', 'circle_radius', str(case['circle_radius']))
        if 'square_width' in case: env_config.set('sim', 'square_width', str(case['square_width']))

        try:
            policy_class = policy_factory[args.policy]
            policy = policy_class(policy_config)
        except:
            policy = policy_class()

        if args.weights == 'latest':
            files = [f for f in os.listdir(args.model_dir) if re.match(r'rl_model_ep\d+\.pth', f)]
            if files:
                files.sort(key=lambda x: os.path.getmtime(os.path.join(args.model_dir, x)))
                weights_path = os.path.join(args.model_dir, files[-1])
            else:
                weights_path = os.path.join(args.model_dir, 'rl_model.pth')
        else:
            weights_path = args.weights

        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('policy_state', checkpoint.get('value_state', checkpoint.get('model', checkpoint)))
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        if hasattr(policy, 'load_state_dict'): policy.load_state_dict(state_dict)
        elif hasattr(policy, 'model'): policy.model.load_state_dict(state_dict)

        if hasattr(policy, 'to'): policy.to(device)
        if hasattr(policy, 'eval'): policy.eval()
        elif hasattr(policy, 'model'): policy.model.eval()

        env = CrowdSim()
        env.configure(env_config)
        env.phase = 'test'
        robot = Robot(env_config, 'robot')
        robot.set_policy(policy)
        robot.env = env
        env.set_robot(robot)

        if hasattr(policy, 'set_env'): policy.set_env(env)
        if hasattr(policy, 'set_phase'): policy.set_phase('test')
        if hasattr(policy, 'set_env_dt'): policy.set_env_dt(0.25)

        success_count, collision_count, timeout_count = 0, 0, 0

        pbar = tqdm(range(args.episodes), ncols=100)
        for ep in pbar:
            episode_seed = random.randint(0, 2**31 - 1)
            outcome = test_episode(env, robot, policy, episode_seed)

            if outcome == 'success': success_count += 1
            elif outcome == 'collision': collision_count += 1
            else: timeout_count += 1

            pbar.set_postfix({'S': f'{success_count}/{ep+1}', 'C': f'{collision_count}/{ep+1}'})

        print(f"\nResults for {case['desc']}:")
        print(f"  SUCCESS:   {success_count}/{args.episodes} ({100*success_count/args.episodes:.1f}%)")
        print(f"  COLLISION: {collision_count}/{args.episodes} ({100*collision_count/args.episodes:.1f}%)")
        print(f"  TIMEOUT:   {timeout_count}/{args.episodes} ({100*timeout_count/args.episodes:.1f}%)")

if __name__ == '__main__':
    main()
