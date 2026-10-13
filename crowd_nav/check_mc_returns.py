#!/usr/bin/env python3
"""检查IL数据集中MC return的实际分布"""
import torch
import numpy as np

# 加载IL数据集
dataset_path = '/root/workspace/nav_data/mamba/camrl/CrowdNav/data/il_dataset_diverse_v5.0.pth'
print(f"Loading dataset from {dataset_path}")
data = torch.load(dataset_path, map_location='cpu', weights_only=False)

print(f"\nDataset keys: {data.keys()}")
print(f"Number of trajectories: {len(data['trajectories'])}")
print(f"Success rate: {data.get('success_rate', 'N/A')}")

# 计算MC returns
gamma = 0.97
all_returns = []
all_rewards_sum = []
traj_lengths = []

for traj, info in data['trajectories'][:1000]:  # 检查前1000条
    states, actions, rewards = traj
    T = len(rewards)
    traj_lengths.append(T)

    # 计算MC return (从第一步看整个轨迹)
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    all_returns.append(G)
    all_rewards_sum.append(sum(rewards))

all_returns = np.array(all_returns)
all_rewards_sum = np.array(all_rewards_sum)
traj_lengths = np.array(traj_lengths)

print(f"\n{'='*60}")
print(f"MC Returns 统计 (gamma={gamma}):")
print(f"{'='*60}")
print(f"Mean:   {all_returns.mean():.4f}")
print(f"Median: {np.median(all_returns):.4f}")
print(f"Std:    {all_returns.std():.4f}")
print(f"Min:    {all_returns.min():.4f}")
print(f"Max:    {all_returns.max():.4f}")
print(f"5%:     {np.percentile(all_returns, 5):.4f}")
print(f"95%:    {np.percentile(all_returns, 95):.4f}")

print(f"\n{'='*60}")
print(f"Rewards Sum 统计:")
print(f"{'='*60}")
print(f"Mean:   {all_rewards_sum.mean():.4f}")
print(f"Median: {np.median(all_rewards_sum):.4f}")
print(f"Min:    {all_rewards_sum.min():.4f}")
print(f"Max:    {all_rewards_sum.max():.4f}")

print(f"\n{'='*60}")
print(f"Trajectory Lengths:")
print(f"{'='*60}")
print(f"Mean:   {traj_lengths.mean():.1f} steps")
print(f"Median: {np.median(traj_lengths):.1f} steps")
print(f"Min:    {traj_lengths.min()}")
print(f"Max:    {traj_lengths.max()}")

# 检查几条具体轨迹
print(f"\n{'='*60}")
print(f"Sample Trajectories:")
print(f"{'='*60}")
for i in range(min(5, len(data['trajectories']))):
    traj, info = data['trajectories'][i]
    states, actions, rewards = traj
    T = len(rewards)

    # MC return
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G

    print(f"\nTraj {i}: len={T}, sum_rewards={sum(rewards):.3f}, MC_return={G:.4f}")
    print(f"  Rewards: {rewards[:5]} ... {rewards[-3:]}")
    print(f"  Info: {info}")
