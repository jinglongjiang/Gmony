#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D9: 空场烟雾测试 (Empty Field Smoke Test)

根据report.md要求：
- 0-1人空场景
- 10回合测试
- 至少7次成功（70%成功率）
- 快速验证基础导航能力
"""

import os
import sys
import logging
import argparse
import configparser
import numpy as np
import torch

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.explorer import Explorer
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.memory import ReplayMemory


class SimpleRandomPolicy:
    """简单的随机策略，用于空场测试"""
    def __init__(self):
        from crowd_nav.contracts import GRID
        self.grid = GRID
        self.num_actions = GRID['n_speeds'] * GRID['n_headings']
        self.multiagent_training = False

    def predict(self, state):
        # 随机选择一个动作
        import random
        action_idx = random.randint(0, self.num_actions - 1)
        from crowd_nav.contracts import discrete_index_to_action
        vx, vy = discrete_index_to_action(action_idx, **self.grid)
        return np.array([vx, vy], dtype=np.float32)

    def set_env(self, env):
        pass

    def set_phase(self, phase):
        pass

    def set_epsilon(self, epsilon):
        pass


def safe_build_policy(policy_key: str, policy_config, device: torch.device):
    """安全构建策略（从train.py复制）"""
    try:
        PolicyClass = policy_factory[policy_key]
        try:
            policy = PolicyClass(policy_config)
        except TypeError:
            policy = PolicyClass()

        if hasattr(policy, 'configure'):
            policy.configure(policy_config)
        if hasattr(policy, 'set_device'):
            policy.set_device(device)
        return policy
    except Exception:
        # Fallback to simple random policy
        return SimpleRandomPolicy()


def setup_logging():
    """设置简单的日志格式"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )


def load_config(path: str) -> configparser.RawConfigParser:
    """加载配置文件"""
    cfg = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    cfg.read(path, encoding='utf-8')
    return cfg


def build_empty_field_env(env_config):
    """构建空场环境（0-1人）"""
    env = CrowdSim()

    # 修改环境配置为空场模式
    env_config.set('sim', 'human_num', '1')  # 0-1人（随机）
    env_config.set('env', 'time_limit', '20')  # 缩短时间限制，快速测试

    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    return env, robot


def run_empty_field_test(policy_path=None, test_rounds=10, success_threshold=0.7):
    """运行空场烟雾测试

    Args:
        policy_path: 策略模型路径，None表示使用随机策略
        test_rounds: 测试回合数（默认10）
        success_threshold: 成功率阈值（默认70%）

    Returns:
        dict: 测试结果统计
    """
    setup_logging()
    logging.info("=" * 50)
    logging.info("空场烟雾测试 (Empty Field Smoke Test)")
    logging.info("=" * 50)

    # 加载配置
    env_config = load_config('configs/env.config')
    policy_config = load_config('configs/policy.config')

    # 构建空场环境
    env, robot = build_empty_field_env(env_config)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")

    # 构建策略
    if policy_path and os.path.exists(policy_path):
        logging.info(f"加载策略模型: {policy_path}")
        try:
            # 加载mamba策略
            policy = safe_build_policy('mamba', policy_config, device)

            # 加载模型权重
            if hasattr(policy, 'load_model'):
                policy.load_model(policy_path)
            else:
                state_dict = torch.load(policy_path, map_location=device)
                policy.model.load_state_dict(state_dict, strict=False)

            policy.set_phase('test')
            policy.set_epsilon(0.0)  # 测试时不使用随机探索

        except Exception as e:
            logging.warning(f"加载策略失败: {e}，使用随机策略")
            policy = SimpleRandomPolicy()
    else:
        logging.info("使用随机基准策略")
        policy = SimpleRandomPolicy()

    # 设置策略绑定
    if hasattr(policy, 'set_env'):
        policy.set_env(env)
    robot.set_policy(policy)

    # 创建Explorer用于运行测试
    memory = ReplayMemory(100)  # 小内存即可
    explorer = Explorer(env, robot, device, memory, gamma=0.95, target_policy=policy)

    # 运行测试
    logging.info(f"开始 {test_rounds} 回合空场测试...")
    logging.info(f"成功率目标: ≥{success_threshold*100:.0f}%")

    results = []
    success_count = 0

    for round_i in range(test_rounds):
        logging.info(f"\n--- 回合 {round_i+1}/{test_rounds} ---")

        try:
            # 运行单个episode
            stats = explorer.run_k_episodes(
                k=1,
                phase='test',
                update_memory=False,
                return_stats=True,
                show_tqdm=False
            )

            success = stats.get('success_rate', 0.0)
            collision = stats.get('collision_rate', 0.0)
            timeout = stats.get('timeout_rate', 0.0)
            nav_time = stats.get('nav_time', 0.0)
            reward = stats.get('total_reward', 0.0)

            results.append({
                'round': round_i + 1,
                'success': success,
                'collision': collision,
                'timeout': timeout,
                'nav_time': nav_time,
                'reward': reward
            })

            if success > 0.5:  # 成功
                success_count += 1
                status = "✓ 成功"
            elif collision > 0.5:  # 碰撞
                status = "✗ 碰撞"
            else:  # 超时
                status = "⏰ 超时"

            logging.info(f"结果: {status} | 导航时间: {nav_time:.1f}s | 奖励: {reward:+.2f}")

        except Exception as e:
            logging.error(f"回合 {round_i+1} 执行失败: {e}")
            results.append({
                'round': round_i + 1,
                'success': 0.0,
                'collision': 1.0,
                'timeout': 0.0,
                'nav_time': 20.0,
                'reward': -10.0
            })

    # 统计结果
    success_rate = success_count / test_rounds
    avg_nav_time = np.mean([r['nav_time'] for r in results if r['success'] > 0.5])
    avg_reward = np.mean([r['reward'] for r in results])

    logging.info("\n" + "=" * 50)
    logging.info("测试结果汇总")
    logging.info("=" * 50)
    logging.info(f"总回合数: {test_rounds}")
    logging.info(f"成功次数: {success_count}")
    logging.info(f"成功率: {success_rate*100:.1f}%")
    logging.info(f"平均导航时间: {avg_nav_time:.1f}s (仅成功回合)")
    logging.info(f"平均奖励: {avg_reward:+.2f}")

    # 判断测试是否通过
    test_passed = success_rate >= success_threshold
    if test_passed:
        logging.info(f"🎉 测试通过！成功率 {success_rate*100:.1f}% ≥ {success_threshold*100:.0f}%")
    else:
        logging.warning(f"❌ 测试失败！成功率 {success_rate*100:.1f}% < {success_threshold*100:.0f}%")

    return {
        'test_passed': test_passed,
        'success_rate': success_rate,
        'success_count': success_count,
        'total_rounds': test_rounds,
        'avg_nav_time': avg_nav_time,
        'avg_reward': avg_reward,
        'results': results
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='空场烟雾测试')
    parser.add_argument('--policy_path', type=str, default=None,
                       help='策略模型路径 (default: None, use ORCA)')
    parser.add_argument('--rounds', type=int, default=10,
                       help='测试回合数 (default: 10)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='成功率阈值 (default: 0.7)')

    args = parser.parse_args()

    # 运行测试
    result = run_empty_field_test(
        policy_path=args.policy_path,
        test_rounds=args.rounds,
        success_threshold=args.threshold
    )

    # 退出码：测试通过返回0，失败返回1
    exit_code = 0 if result['test_passed'] else 1
    sys.exit(exit_code)


if __name__ == '__main__':
    main()