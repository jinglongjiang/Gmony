#!/usr/bin/env python3
"""检查IL dataset中的meta['source']字段"""
import torch
import os
import glob

# 查找IL数据集
candidates = [
    'data/il_dataset_diverse_v4.0.pth',
    'data/il_dataset_diverse_v3.1.pth',
    '../data/il_dataset_diverse_v3.1.pth',
]

dataset_path = None
for c in candidates:
    if os.path.exists(c):
        dataset_path = c
        break

if not dataset_path:
    # 使用glob搜索
    patterns = ['data/il_dataset*.pth', '../data/il_dataset*.pth']
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            dataset_path = files[0]
            break

if not dataset_path:
    print("❌ 未找到IL数据集")
    exit(1)

print(f"✓ 找到数据集: {dataset_path}")

# 加载并检查
dataset = torch.load(dataset_path, map_location='cpu')

if isinstance(dataset, dict) and 'trajectories' in dataset:
    trajs = dataset['trajectories']
    print(f"✓ 总轨迹数: {len(trajs)}")

    # 检查前5条的meta
    print("\n前5条轨迹的meta['source']字段：")
    for i, traj in enumerate(trajs[:5]):
        meta = traj.get('meta', {})
        source = meta.get('source', '<MISSING>')
        print(f"  [{i}] source = '{source}'")
        if i == 0:
            print(f"      完整meta: {meta}")

    # 统计所有source字段的值
    source_counts = {}
    missing_count = 0
    for traj in trajs:
        meta = traj.get('meta', {})
        source = meta.get('source', None)
        if source is None:
            missing_count += 1
        else:
            source_counts[source] = source_counts.get(source, 0) + 1

    print(f"\nsource字段统计：")
    for k, v in source_counts.items():
        print(f"  '{k}': {v} ({v/len(trajs)*100:.1f}%)")
    if missing_count > 0:
        print(f"  <MISSING>: {missing_count} ({missing_count/len(trajs)*100:.1f}%)")

    print(f"\n结论：")
    if missing_count == len(trajs):
        print("  ✓ 所有轨迹都没有source字段，setdefault('source', 'IL')会正常工作")
    elif 'IL' in source_counts and source_counts['IL'] == len(trajs):
        print("  ✓ 所有轨迹的source='IL'，采样逻辑应该正常")
    else:
        print("  ❌ 轨迹的source字段值不一致，这会导致IL采样失败！")
else:
    print("❌ 数据集格式不符合预期")
