#!/usr/bin/env python3
"""
BC训练深度诊断脚本
分析为什么value-guided BC仍然只有30%成功率
"""

import torch
import numpy as np
import pickle
from pathlib import Path
import sys

def analyze_quantization_error(data_path):
    """诊断1: 分析ORCA动作量化误差"""
    print("\n" + "="*60)
    print("诊断1: ORCA动作量化误差分析")
    print("="*60)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # 提取专家动作
    expert_actions = []  # (vx, vy)连续动作
    quantized_actions = []  # 量化后的离散索引

    for traj in data:
        for transition in traj:
            state, action_tuple = transition[0], transition[1]
            # action_tuple: (action_idx, vx, vy, ...)
            action_idx = action_tuple[0]
            vx, vy = action_tuple[1], action_tuple[2]

            expert_actions.append([vx, vy])
            quantized_actions.append(action_idx)

    expert_actions = np.array(expert_actions)
    quantized_actions = np.array(quantized_actions)

    # 计算速度和方向
    expert_speeds = np.sqrt(expert_actions[:, 0]**2 + expert_actions[:, 1]**2)
    expert_headings = np.arctan2(expert_actions[:, 1], expert_actions[:, 0]) * 180 / np.pi

    # 反量化：从动作索引还原速度和方向
    n_speeds = 5
    n_headings = 16
    speed_bins = [0.0, 0.25, 0.5, 0.75, 1.0]

    speed_indices = quantized_actions // n_headings
    heading_indices = quantized_actions % n_headings

    quantized_speeds = np.array([speed_bins[min(si, len(speed_bins)-1)] for si in speed_indices])
    quantized_headings = heading_indices * (360.0 / n_headings)

    # 计算误差
    speed_errors = np.abs(expert_speeds - quantized_speeds)
    heading_errors = np.minimum(
        np.abs(expert_headings - quantized_headings),
        360 - np.abs(expert_headings - quantized_headings)
    )

    print(f"\n📊 量化误差统计 (共{len(expert_actions)}个样本):")
    print(f"\n速度量化误差:")
    print(f"  平均误差: {speed_errors.mean():.4f} m/s")
    print(f"  中位误差: {np.median(speed_errors):.4f} m/s")
    print(f"  最大误差: {speed_errors.max():.4f} m/s")
    print(f"  误差>0.1的比例: {(speed_errors > 0.1).mean()*100:.1f}%")

    print(f"\n方向量化误差:")
    print(f"  平均误差: {heading_errors.mean():.2f}°")
    print(f"  中位误差: {np.median(heading_errors):.2f}°")
    print(f"  最大误差: {heading_errors.max():.2f}°")
    print(f"  误差>10°的比例: {(heading_errors > 10).mean()*100:.1f}%")

    # 分析ORCA速度分布
    print(f"\n📊 ORCA速度分布 vs 速度档位:")
    print(f"  速度档位: {speed_bins}")
    print(f"\nORCA速度统计:")
    print(f"  [0.00, 0.125): {((expert_speeds >= 0.00) & (expert_speeds < 0.125)).sum()} ({((expert_speeds >= 0.00) & (expert_speeds < 0.125)).mean()*100:.1f}%)")
    print(f"  [0.125, 0.375): {((expert_speeds >= 0.125) & (expert_speeds < 0.375)).sum()} ({((expert_speeds >= 0.125) & (expert_speeds < 0.375)).mean()*100:.1f}%)")
    print(f"  [0.375, 0.625): {((expert_speeds >= 0.375) & (expert_speeds < 0.625)).sum()} ({((expert_speeds >= 0.375) & (expert_speeds < 0.625)).mean()*100:.1f}%)")
    print(f"  [0.625, 0.875): {((expert_speeds >= 0.625) & (expert_speeds < 0.875)).sum()} ({((expert_speeds >= 0.625) & (expert_seeds < 0.875)).mean()*100:.1f}%)")
    print(f"  [0.875, 1.00]: {((expert_speeds >= 0.875) & (expert_speeds <= 1.00)).sum()} ({((expert_speeds >= 0.875) & (expert_speeds <= 1.00)).mean()*100:.1f}%)")

    print(f"\n关键发现:")
    if (expert_speeds < 0.3).mean() > 0.3:
        print(f"  ⚠️  {(expert_speeds < 0.3).mean()*100:.1f}% 的ORCA动作速度<0.3 (慢速/停顿)")
    if speed_errors.mean() > 0.1:
        print(f"  ⚠️  平均速度误差{speed_errors.mean():.3f}较大，量化损失严重")
    if heading_errors.mean() > 15:
        print(f"  ⚠️  平均方向误差{heading_errors.mean():.1f}°较大，16方向分辨率不足")

    return {
        'speed_error_mean': speed_errors.mean(),
        'heading_error_mean': heading_errors.mean(),
        'expert_speeds': expert_speeds,
        'quantized_speeds': quantized_speeds,
        'n_samples': len(expert_actions)
    }


def analyze_bc_training_data(data_path):
    """诊断2: 分析BC训练数据质量"""
    print("\n" + "="*60)
    print("诊断2: BC训练数据质量分析")
    print("="*60)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # 统计轨迹结果
    n_success = 0
    n_collision = 0
    n_timeout = 0

    success_returns = []
    collision_returns = []
    timeout_returns = []

    for traj in data:
        # 最后一个transition的reward反映episode结果
        final_reward = traj[-1][2]  # (state, action, reward, done)

        # 计算轨迹回报
        traj_return = sum([t[2] for t in traj])

        if final_reward > 0.5:  # success_reward = 1.0
            n_success += 1
            success_returns.append(traj_return)
        elif final_reward < -0.1:  # collision_penalty = -0.25
            n_collision += 1
            collision_returns.append(traj_return)
        else:
            n_timeout += 1
            timeout_returns.append(traj_return)

    total = len(data)
    print(f"\n📊 轨迹结果统计 (共{total}条轨迹):")
    print(f"  成功: {n_success} ({n_success/total*100:.1f}%)")
    print(f"  碰撞: {n_collision} ({n_collision/total*100:.1f}%)")
    print(f"  超时: {n_timeout} ({n_timeout/total*100:.1f}%)")

    print(f"\n📊 轨迹回报分布:")
    if success_returns:
        print(f"  成功轨迹: 均值={np.mean(success_returns):.3f}, 中位={np.median(success_returns):.3f}")
    if collision_returns:
        print(f"  碰撞轨迹: 均值={np.mean(collision_returns):.3f}, 中位={np.median(collision_returns):.3f}")
    if timeout_returns:
        print(f"  超时轨迹: 均值={np.mean(timeout_returns):.3f}, 中位={np.median(timeout_returns):.3f}")

    # 分析value权重分布（模拟训练时的计算）
    all_returns = []
    for traj in data:
        gamma = 0.97
        traj_return = 0
        for i in range(len(traj) - 1, -1, -1):
            reward = traj[i][2]
            traj_return = reward + gamma * traj_return
            all_returns.append(traj_return)

    all_returns = np.array(all_returns)
    value_weights = 1.0 / (1.0 + np.exp(-(all_returns - all_returns.mean()) / (all_returns.std() + 1e-6)))

    print(f"\n📊 Value权重分布 (sigmoid归一化):")
    print(f"  均值: {value_weights.mean():.3f}")
    print(f"  标准差: {value_weights.std():.3f}")
    print(f"  最小值: {value_weights.min():.3f}")
    print(f"  最大值: {value_weights.max():.3f}")
    print(f"  中位数: {np.median(value_weights):.3f}")

    print(f"\n关键发现:")
    if n_success / total < 0.5:
        print(f"  ⚠️  ORCA成功率仅{n_success/total*100:.1f}%，专家演示质量不足")
    if value_weights.std() < 0.1:
        print(f"  ⚠️  Value权重标准差仅{value_weights.std():.3f}，加权效果有限")
    if abs(value_weights.mean() - 0.5) < 0.05:
        print(f"  ⚠️  Value权重均值{value_weights.mean():.3f}≈0.5，sigmoid归一化失效")

    return {
        'success_rate': n_success / total,
        'collision_rate': n_collision / total,
        'value_weight_std': value_weights.std(),
        'n_trajectories': total
    }


def compare_with_sarl():
    """诊断3: 对比SARL的训练方式"""
    print("\n" + "="*60)
    print("诊断3: SARL vs Mamba BC训练方式对比")
    print("="*60)

    print("\n📚 SARL的成功之道:")
    print("  1. ✅ 只训练value network (MSE loss)")
    print("  2. ✅ 学习目标: V(s) = MC_return")
    print("  3. ✅ 测试时: one-step lookahead选动作")
    print("  4. ✅ 无量化误差（训练时不涉及动作预测）")

    print("\n📚 我们的Mamba BC:")
    print("  1. ❌ 训练policy network (CrossEntropy loss)")
    print("  2. ❌ 学习目标: argmax P(a|s) = quantized_ORCA_action")
    print("  3. ❌ 测试时: 直接输出policy_head的argmax")
    print("  4. ❌ 严重量化误差（ORCA连续→80离散）")

    print("\n🔍 核心差异:")
    print("  SARL: 学习'状态价值' → 运行时选最优离散动作")
    print("  Mamba BC: 学习'动作分类' → 被迫学习量化后的坏动作")

    print("\n💡 可能的解决方案:")
    print("  方案A: 切换到纯value训练（放弃BC，模仿SARL）")
    print("         - 只训练value_head，测试时one-step lookahead")
    print("         - 问题：用户说'Mamba RL需要离散动作'")
    print("  ")
    print("  方案B: Soft BC标签（缓解量化误差）")
    print("         - 不用硬标签action_idx，用软分布")
    print("         - 给临近动作分配概率（基于距离）")
    print("         - 例：ORCA=(0.35, 45°) → 给(0.25,45°)权重0.6, (0.5,45°)权重0.4")
    print("  ")
    print("  方案C: 增加动作分辨率")
    print("         - 用户已拒绝（破坏checkpoint兼容性）")


def main():
    """主诊断流程"""
    print("\n" + "🔬"*30)
    print("BC训练深度诊断 - 找出30%成功率的根本原因")
    print("🔬"*30)

    # 查找IL数据文件
    data_dir = Path("data")
    il_files = list(data_dir.glob("**/il_episodes_*.pkl"))

    if not il_files:
        print("\n❌ 错误: 未找到IL数据文件 (data/**/il_episodes_*.pkl)")
        print("请先运行IL训练生成数据")
        return

    # 使用最新的IL数据
    il_data_path = sorted(il_files, key=lambda p: p.stat().st_mtime)[-1]
    print(f"\n📁 使用IL数据: {il_data_path}")
    print(f"   文件大小: {il_data_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 执行诊断
    results = {}

    try:
        results['quantization'] = analyze_quantization_error(il_data_path)
    except Exception as e:
        print(f"\n❌ 量化误差分析失败: {e}")

    try:
        results['data_quality'] = analyze_bc_training_data(il_data_path)
    except Exception as e:
        print(f"\n❌ 数据质量分析失败: {e}")

    compare_with_sarl()

    # 综合诊断结论
    print("\n" + "="*60)
    print("🎯 综合诊断结论")
    print("="*60)

    if 'quantization' in results:
        q = results['quantization']
        print(f"\n量化误差: 速度±{q['speed_error_mean']:.3f}m/s, 方向±{q['heading_error_mean']:.1f}°")

    if 'data_quality' in results:
        d = results['data_quality']
        print(f"数据质量: ORCA成功率{d['success_rate']*100:.1f}%, value权重std={d['value_weight_std']:.3f}")

    print("\n🔍 根本问题:")
    print("  BC训练 ≠ SARL训练")
    print("  我们在强迫网络学习'量化后的ORCA动作'")
    print("  SARL是学习'状态好坏'，运行时自己选动作")
    print("  量化误差 + 硬标签 = BC学到的是损坏的专家演示")

    print("\n💊 建议修复方案:")
    print("  推荐: Soft BC Labels（温和改进，兼容现有结构）")
    print("  备选: 切换到纯Value训练（激进改动，需验证RL阶段）")


if __name__ == "__main__":
    main()
