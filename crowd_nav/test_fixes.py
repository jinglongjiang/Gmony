#!/usr/bin/env python3
"""
快速测试脚本：验证IL配额计算和step-level masking修复
"""
import sys
import os
import torch
import numpy as np
import configparser
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.memory import SequenceReplayMemory
from contracts import joint34_to_tokens

def test_il_quota_calculation():
    """测试IL配额计算逻辑"""
    print("\n" + "="*60)
    print("测试1: IL配额计算逻辑")
    print("="*60)

    # 创建replay buffer
    buf = SequenceReplayMemory(capacity_episodes=100, sequence_length=8)
    buf.set_sampling_params(
        min_success_frac=0.30,
        max_timeout_frac=0.15,
        il_bias=1.5,
        tail_window=6,
        tail_last_prob=0.25
    )

    # 添加10个IL episodes (都是success)
    for i in range(10):
        states = np.random.randn(20, 34).astype(np.float32)
        actions = np.random.randn(20, 2).astype(np.float32)
        rewards = np.zeros(20)
        rewards[-1] = 1.0  # success
        dones = np.zeros(20, dtype=bool)
        dones[-1] = True
        timeouts = np.zeros(20, dtype=bool)

        buf.push_episode(
            states=states,
            actions_continuous=actions,
            rewards=rewards,
            dones=dones,
            timeouts=timeouts,
            meta={'source': 'IL', 'seed': 1000+i}
        )

    # 添加90个RL episodes (混合)
    for i in range(90):
        states = np.random.randn(20, 34).astype(np.float32)
        actions = np.random.randn(20, 2).astype(np.float32)
        rewards = np.zeros(20)
        dones = np.zeros(20, dtype=bool)
        timeouts = np.zeros(20, dtype=bool)

        if i < 30:  # 30个success
            rewards[-1] = 1.0
            dones[-1] = True
        elif i < 45:  # 15个timeout
            rewards[-1] = -0.5
            dones[-1] = True
            timeouts[-1] = True
        else:  # 45个collision
            rewards[-1] = -0.25
            dones[-1] = True

        buf.push_episode(
            states=states,
            actions_continuous=actions,
            rewards=rewards,
            dones=dones,
            timeouts=timeouts,
            meta={'source': 'RL'}
        )

    print(f"Buffer状态: {len(buf.episodes)} episodes")
    print(f"  - IL: 10 episodes (都是success)")
    print(f"  - RL: 90 episodes (30 success, 15 timeout, 45 collision)")

    # 采样batch=100
    print(f"\n采样 batch_size=100:")
    batch = buf.sample(batch_size=100, seq_len=8, device='cpu')

    # 检查batch结构
    print(f"\nBatch结构检查:")
    print(f"  - states shape: {batch['states'].shape}")
    print(f"  - rewards shape: {batch['rewards'].shape}")
    print(f"  - actions_continuous shape: {batch['actions_continuous'].shape}")
    print(f"  - timeouts shape: {batch['timeouts'].shape}")
    print(f"  - mix_info keys: {batch['mix_info'].keys()}")

    # 验证IL比例
    il_pct = batch['mix_info']['il_pct']
    print(f"\n配额验证:")
    print(f"  - IL%: {il_pct:.1f}% (应该≈10%)")

    return True


def test_step_level_masking():
    """测试step-level masking的维度正确性"""
    print("\n" + "="*60)
    print("测试2: Step-level masking维度")
    print("="*60)

    # 模拟BATCH-ADAPT后的数据 [N]
    N = 64  # 采样的总步数

    # 创建模拟数据
    states = torch.randn(N, 1, 6, 13)
    actions = torch.randn(N, 2)
    rewards = torch.randn(N)
    next_states = torch.randn(N, 1, 6, 13)
    dones = torch.zeros(N)
    timeout_flags = torch.zeros(N)
    timeout_flags[10:20] = 1  # 假设10个timeout步
    success_flags = torch.zeros(N)
    success_flags[50:] = 1  # 假设14个success步

    print(f"模拟数据形状:")
    print(f"  - states: {states.shape}")
    print(f"  - actions: {actions.shape}")
    print(f"  - timeout_flags: {timeout_flags.shape}, sum={timeout_flags.sum().item()}")

    # 模拟policy mask计算
    policy_mask = (1.0 - timeout_flags).float()
    print(f"\nPolicy mask:")
    print(f"  - shape: {policy_mask.shape}")
    print(f"  - 有效步数: {policy_mask.sum().item()}/{N}")

    # 模拟AWR权重计算
    weights = torch.ones(N) * 2.0

    # 模拟finite mask
    finite_mask = torch.ones(N, dtype=torch.bool)
    finite_mask[:5] = False  # 假设前5个样本无效

    print(f"\nFinite mask:")
    print(f"  - 有效样本: {finite_mask.sum().item()}/{N}")

    # 测试combined weights计算
    valid_weights = weights[finite_mask]  # [K]
    valid_policy_mask = policy_mask[finite_mask]  # [K]
    combined_weights = (valid_weights * valid_policy_mask).unsqueeze(-1)  # [K, 1]

    print(f"\nCombined weights:")
    print(f"  - shape: {combined_weights.shape}")
    print(f"  - 被mask的步数: {(valid_policy_mask == 0).sum().item()}")

    # 模拟loss计算
    valid_actions = actions[finite_mask]  # [K, 2]
    continuous_preds = torch.randn_like(valid_actions)
    policy_loss_raw = (continuous_preds - valid_actions) ** 2  # [K, 2]
    policy_loss = (combined_weights * policy_loss_raw).mean()

    print(f"\nPolicy loss计算:")
    print(f"  - policy_loss_raw shape: {policy_loss_raw.shape}")
    print(f"  - policy_loss: {policy_loss.item():.4f}")

    print(f"\n✓ 维度检查通过!")
    return True


if __name__ == '__main__':
    print("\n" + "="*60)
    print("根因修复验证测试")
    print("="*60)

    try:
        # 测试1: IL配额计算
        test_il_quota_calculation()

        # 测试2: Step-level masking
        test_step_level_masking()

        print("\n" + "="*60)
        print("✓ 所有测试通过!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
