#!/usr/bin/env python3
"""
完整测试：验证从数据存储到Buffer采样的4D流程
确保统一使用 [B, T, 6, 13] 格式
"""
import sys
import numpy as np
import torch
sys.path.insert(0, '/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')

from crowd_nav.utils.memory import SequenceReplayMemory

def test_new_data_storage():
    """测试新数据存储流程：34D → [T,6,13] → Buffer → [B,T,6,13]"""
    print("=" * 60)
    print("测试1: 新RL数据存储流程 (34D raw states)")
    print("=" * 60)

    memory = SequenceReplayMemory(capacity_episodes=1000, sequence_length=12)

    # 模拟RL rollout收集的数据 (34D raw states)
    T = 20
    states_34d = np.random.randn(T, 34).astype(np.float32)
    rewards = np.random.randn(T).astype(np.float32)
    dones = np.zeros(T, dtype=np.bool_)
    dones[-1] = True
    timeouts = np.zeros(T, dtype=np.bool_)
    actions = np.random.randn(T, 2).astype(np.float32)

    # 存储episode
    memory.push_episode(
        states=states_34d,
        rewards=rewards,
        dones=dones,
        timeouts=timeouts,
        actions_continuous=actions,
        meta={'event': 'success'}
    )

    # 检查存储的tokens格式
    ep = memory.episodes[0]
    print(f"1. 存储的tokens shape: {ep['tokens'].shape}")

    if ep['tokens'].ndim == 3 and ep['tokens'].shape[1:] == (6, 13):
        print(f"   ✅ 正确: [T={ep['tokens'].shape[0]}, 6, 13] - 3D without channel")
    else:
        print(f"   ❌ 错误: {ep['tokens'].shape}")
        return False

    # 从buffer采样
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = memory.sample(batch_size=4, seq_len=12, device=device)

    print(f"2. Buffer采样 states shape: {batch['states'].shape}")

    if batch['states'].dim() == 4 and batch['states'].shape[2:] == (6, 13):
        print(f"   ✅ 正确: [B={batch['states'].shape[0]}, T={batch['states'].shape[1]}, 6, 13]")
    else:
        print(f"   ❌ 错误: {batch['states'].shape}")
        return False

    print("\n✅ 新数据存储流程测试通过！\n")
    return True

def test_old_data_compatibility():
    """测试旧数据兼容性：[T,1,6,13] → Buffer → [B,T,6,13]"""
    print("=" * 60)
    print("测试2: 旧IL数据兼容性 (已有的[T,1,6,13]格式)")
    print("=" * 60)

    memory = SequenceReplayMemory(capacity_episodes=1000, sequence_length=12)

    # 模拟旧IL数据 (已经是token格式，带channel维度)
    T = 20
    states_tokens = np.random.randn(T, 1, 6, 13).astype(np.float32)  # 旧格式
    rewards = np.random.randn(T).astype(np.float32)
    dones = np.zeros(T, dtype=np.bool_)
    dones[-1] = True
    timeouts = np.zeros(T, dtype=np.bool_)
    actions = np.random.randn(T, 2).astype(np.float32)

    # 存储episode
    memory.push_episode(
        states=states_tokens,
        rewards=rewards,
        dones=dones,
        timeouts=timeouts,
        actions_continuous=actions,
        meta={'event': 'success'}
    )

    # 检查存储的tokens格式
    ep = memory.episodes[0]
    print(f"1. 存储的tokens shape: {ep['tokens'].shape}")

    if ep['tokens'].ndim == 3 and ep['tokens'].shape[1:] == (6, 13):
        print(f"   ✅ 正确: [T={ep['tokens'].shape[0]}, 6, 13] - squeeze成功")
    else:
        print(f"   ❌ 错误: {ep['tokens'].shape}")
        return False

    # 从buffer采样
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = memory.sample(batch_size=4, seq_len=12, device=device)

    print(f"2. Buffer采样 states shape: {batch['states'].shape}")

    if batch['states'].dim() == 4 and batch['states'].shape[2:] == (6, 13):
        print(f"   ✅ 正确: [B={batch['states'].shape[0]}, T={batch['states'].shape[1]}, 6, 13]")
    else:
        print(f"   ❌ 错误: {batch['states'].shape}")
        return False

    print("\n✅ 旧数据兼容性测试通过！\n")
    return True

def test_token_format():
    """测试新token格式：[T,6,13]格式"""
    print("=" * 60)
    print("测试3: 新token格式 ([T,6,13]直接输入)")
    print("=" * 60)

    memory = SequenceReplayMemory(capacity_episodes=1000, sequence_length=12)

    # 模拟新token格式数据 (无channel维度)
    T = 20
    states_tokens = np.random.randn(T, 6, 13).astype(np.float32)  # 新正确格式
    rewards = np.random.randn(T).astype(np.float32)
    dones = np.zeros(T, dtype=np.bool_)
    dones[-1] = True
    timeouts = np.zeros(T, dtype=np.bool_)
    actions = np.random.randn(T, 2).astype(np.float32)

    # 存储episode
    memory.push_episode(
        states=states_tokens,
        rewards=rewards,
        dones=dones,
        timeouts=timeouts,
        actions_continuous=actions,
        meta={'event': 'success'}
    )

    # 检查存储的tokens格式
    ep = memory.episodes[0]
    print(f"1. 存储的tokens shape: {ep['tokens'].shape}")

    if ep['tokens'].ndim == 3 and ep['tokens'].shape[1:] == (6, 13):
        print(f"   ✅ 正确: [T={ep['tokens'].shape[0]}, 6, 13] - 保持不变")
    else:
        print(f"   ❌ 错误: {ep['tokens'].shape}")
        return False

    # 从buffer采样
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = memory.sample(batch_size=4, seq_len=12, device=device)

    print(f"2. Buffer采样 states shape: {batch['states'].shape}")

    if batch['states'].dim() == 4 and batch['states'].shape[2:] == (6, 13):
        print(f"   ✅ 正确: [B={batch['states'].shape[0]}, T={batch['states'].shape[1]}, 6, 13]")
    else:
        print(f"   ❌ 错误: {batch['states'].shape}")
        return False

    # 验证可以直接用于Spatial Encoder
    states = batch['states']
    robot_tokens = states[:, :, 0, :]
    humans_all = states[:, :, 3:6, :]

    print(f"3. Spatial Encoder提取:")
    print(f"   - robot_tokens: {robot_tokens.shape} (应为[B,T,13])")
    print(f"   - humans_all: {humans_all.shape} (应为[B,T,3,13])")

    if robot_tokens.shape[-1] == 13 and humans_all.shape[-2:] == (3, 13):
        print(f"   ✅ 正确: 可以直接用于Spatial Encoder")
    else:
        print(f"   ❌ 错误: 维度不匹配")
        return False

    print("\n✅ 新token格式测试通过！\n")
    return True

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("完整4D流程测试")
    print("验证: Episode存储 → Buffer采样 → Spatial Encoder")
    print("=" * 60 + "\n")

    try:
        # 测试1: 新RL数据
        test1 = test_new_data_storage()

        # 测试2: 旧IL数据兼容
        test2 = test_old_data_compatibility()

        # 测试3: 新token格式
        test3 = test_token_format()

        if test1 and test2 and test3:
            print("=" * 60)
            print("🎉 所有测试通过！4D统一格式正确！")
            print("=" * 60)
            print("\n关键点:")
            print("1. 新RL数据 (34D) → 存储为 [T,6,13]")
            print("2. 旧IL数据 ([T,1,6,13]) → squeeze为 [T,6,13]")
            print("3. 新token ([T,6,13]) → 保持不变")
            print("4. Buffer采样 → 统一返回 [B,T,6,13]")
            print("5. Spatial Encoder → 正确接收4D输入")
            print("\n可以同步到服务器测试！")
            return 0
        else:
            print("=" * 60)
            print("❌ 部分测试失败！")
            print("=" * 60)
            return 1
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
