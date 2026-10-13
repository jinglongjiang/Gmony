#!/usr/bin/env python3
"""测试T_last的实际形状"""
import torch

# 模拟train.py中的构造过程
B, T = 10, 8  # 10个episodes，每个8步
TOUT = torch.zeros(B, T)  # [B, T]

# 模拟BATCH-ADAPT的采样过程
T_sel = []
for b in range(B):
    for t in [0, 3, 7]:  # 每个episode采样3步
        T_sel.append(TOUT[b, t].view(1))  # 标量.view(1) = [1]

# 拼接
T_last = torch.cat(T_sel, dim=0)

print(f"TOUT shape: {TOUT.shape}")
print(f"TOUT[0,0] shape: {TOUT[0,0].shape}")
print(f"TOUT[0,0].view(1) shape: {TOUT[0,0].view(1).shape}")
print(f"len(T_sel): {len(T_sel)}")
print(f"T_sel[0] shape: {T_sel[0].shape}")
print(f"T_last shape: {T_last.shape}")
print(f"T_last dim: {T_last.dim()}")

# 测试policy_mask计算
policy_mask = (1.0 - T_last).float()
print(f"\npolicy_mask shape: {policy_mask.shape}")
print(f"policy_mask dim: {policy_mask.dim()}")

# 测试finite_mask索引
finite_mask = torch.ones(T_last.shape[0], dtype=torch.bool)
finite_mask[:5] = False
valid_policy_mask = policy_mask[finite_mask]
print(f"\nvalid_policy_mask shape: {valid_policy_mask.shape}")

# 测试combined_weights
valid_weights = torch.ones(finite_mask.sum().item())
combined_weights = (valid_weights * valid_policy_mask).unsqueeze(-1)
print(f"combined_weights shape: {combined_weights.shape}")

# 测试policy_loss计算
valid_actions = torch.randn(finite_mask.sum().item(), 2)
continuous_preds = torch.randn_like(valid_actions)
policy_loss_raw = (continuous_preds - valid_actions) ** 2
print(f"\npolicy_loss_raw shape: {policy_loss_raw.shape}")
print(f"combined_weights * policy_loss_raw shape: {(combined_weights * policy_loss_raw).shape}")
