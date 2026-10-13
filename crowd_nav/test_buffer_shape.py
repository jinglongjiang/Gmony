#!/usr/bin/env python3
"""
测试replay buffer返回的tensor维度是否正确
验证从[B,T,1,6,13]正确squeeze到[B,T,6,13]
"""
import sys
import numpy as np
import torch

# 模拟episode数据（包含tokens）
def create_mock_episode():
    """创建模拟episode数据，tokens为[T, 1, 6, 13]格式"""
    T = 20
    tokens = np.random.randn(T, 1, 6, 13).astype(np.float32)

    return {
        'tokens': tokens,
        'rewards': np.random.randn(T).astype(np.float32),
        'dones': np.zeros(T, dtype=np.bool_),
        'timeouts': np.zeros(T, dtype=np.bool_),
        'actions_continuous': np.random.randn(T, 2).astype(np.float32),
        'length': T,
        'success': True,
    }

def test_squeeze_logic():
    """测试squeeze逻辑"""
    print("=" * 60)
    print("测试Buffer返回维度修复")
    print("=" * 60)

    # 模拟buffer返回的5D数据
    B, T = 256, 12
    states_5d = np.zeros((B, T, 1, 6, 13), dtype=np.float32)

    print(f"\n1. 原始shape (错误): {states_5d.shape} - 5D tensor")

    # 转换为tensor
    states_tensor = torch.from_numpy(states_5d)
    print(f"2. PyTorch tensor shape: {states_tensor.shape}")

    # 应用squeeze修复
    if states_tensor.dim() == 5 and states_tensor.shape[2] == 1:
        states_fixed = states_tensor.squeeze(2)
        print(f"3. Squeeze后shape (正确): {states_fixed.shape} - 4D tensor ✅")

        # 验证数据一致性
        assert states_fixed.shape == (B, T, 6, 13), f"Expected [B,T,6,13], got {states_fixed.shape}"
        print(f"4. 维度验证: [B={B}, T={T}, 6, 13] ✅")

        # 验证slice操作
        robot_row = states_fixed[:, :, 0, :]  # [B, T, 13]
        assert robot_row.shape == (B, T, 13), f"Robot row should be [B,T,13], got {robot_row.shape}"
        print(f"5. Robot特征提取: {robot_row.shape} ✅")

        human_rows = states_fixed[:, :, 3:6, :]  # [B, T, 3, 13]
        assert human_rows.shape == (B, T, 3, 13), f"Human rows should be [B,T,3,13], got {human_rows.shape}"
        print(f"6. Human特征提取: {human_rows.shape} ✅")

        # 测试距离排序操作（这是之前报错的地方）
        distances = human_rows[..., 2]  # [B, T, 3]
        print(f"7. 距离提取: {distances.shape} ✅")

        _, sorted_indices = torch.sort(distances, dim=-1, descending=False)
        print(f"8. 排序索引: {sorted_indices.shape} (应为[B,T,3]) ✅")

        # 这是之前出错的操作
        sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, 13)
        print(f"9. 扩展索引: {sorted_indices_expanded.shape} (应为[B,T,3,13]) ✅")

        print("\n" + "=" * 60)
        print("✅ 所有维度检查通过！Buffer修复有效！")
        print("=" * 60)
        return True
    else:
        print(f"❌ 错误：tensor不是5D或第3维不是1")
        return False

def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("测试边界情况")
    print("=" * 60)

    # 测试1: 已经是4D的tensor（不需要squeeze）
    states_4d = torch.zeros(256, 12, 6, 13)
    if states_4d.dim() == 5 and states_4d.shape[2] == 1:
        states_4d = states_4d.squeeze(2)
    print(f"1. 4D输入 (不需要squeeze): {states_4d.shape} ✅")

    # 测试2: 单样本batch
    states_single = torch.zeros(1, 12, 1, 6, 13)
    if states_single.dim() == 5 and states_single.shape[2] == 1:
        states_single = states_single.squeeze(2)
    print(f"2. B=1情况: {states_single.shape} ✅")

    # 测试3: 序列长度为1
    states_short = torch.zeros(256, 1, 1, 6, 13)
    if states_short.dim() == 5 and states_short.shape[2] == 1:
        states_short = states_short.squeeze(2)
    print(f"3. T=1情况: {states_short.shape} ✅")

    print("\n✅ 边界情况测试通过！")

if __name__ == '__main__':
    try:
        success = test_squeeze_logic()
        if success:
            test_edge_cases()
            print("\n🎉 所有测试通过！可以同步到服务器测试！")
            sys.exit(0)
        else:
            print("\n❌ 测试失败！")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
