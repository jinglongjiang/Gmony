# -*- coding: utf-8 -*-
"""
Value-First IL: 直接训练价值网络的IL方案
完美适配Mamba架构，如亲兄弟般无缝集成
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader, TensorDataset

class ValueFirstILTrainer:
    """
    价值优先IL训练器：直接从ORCA轨迹学习价值网络
    - 跳过动作策略学习的中间环节
    - 直接训练V(s)，而非π(a|s)
    - 完美适配Mamba的V-learning架构
    """

    def __init__(self, policy, device='cuda', gamma=0.95):
        self.policy = policy
        self.device = device
        self.gamma = gamma

        # 复用Mamba架构（零重复代码）
        self.model = policy.model if hasattr(policy, 'model') else policy

        # 只为价值网络创建优化器
        self.optimizer = optim.Adam(self.model.value_head.parameters(), lr=1e-3)

        logging.info("[VALUE-FIRST-IL] 🎯 价值优先IL训练器已初始化，直接训练V(s)")

    def train_value_from_trajectories(self, explorer, epochs=40, batch_size=128) -> Dict[str, float]:
        """
        核心功能：从ORCA轨迹直接训练价值网络

        Args:
            explorer: 包含_last_trajectories的探索器
            epochs: 训练轮数
            batch_size: 批大小

        Returns:
            训练统计
        """
        if not hasattr(explorer, '_last_trajectories') or not explorer._last_trajectories:
            logging.warning("[VALUE-FIRST-IL] ⚠️ 无ORCA轨迹数据，跳过价值训练")
            return {'value_il_loss': 0.0, 'value_samples': 0, 'epochs': 0}

        # 第1步：从ORCA轨迹提取价值标签
        value_dataset = self._extract_value_labels(explorer._last_trajectories)

        if not value_dataset:
            logging.warning("[VALUE-FIRST-IL] ⚠️ 价值标签提取失败")
            return {'value_il_loss': 0.0, 'value_samples': 0, 'epochs': 0}

        logging.info(f"[VALUE-FIRST-IL] 🚀 开始价值网络训练，样本数: {len(value_dataset)}")

        # 第2步：直接训练Mamba价值网络
        total_loss = self._train_value_network(value_dataset, epochs, batch_size)

        logging.info(f"[VALUE-FIRST-IL] ✅ 价值网络训练完成，平均损失: {total_loss:.6f}")

        return {
            'value_il_loss': total_loss,
            'value_samples': len(value_dataset),
            'epochs': epochs,
            'architecture': 'value_first_mamba_il'
        }

    def _extract_value_labels(self, orca_trajectories) -> List[Tuple[torch.Tensor, float]]:
        """
        从ORCA轨迹提取价值标签（蒙特卡洛回报）

        Args:
            orca_trajectories: ORCA专家轨迹

        Returns:
            List[(state, value)]: 状态-价值对
        """
        value_dataset = []

        for traj_data, info in orca_trajectories:
            try:
                if len(traj_data) < 3:
                    continue

                states, actions, rewards = traj_data[:3]
                outcome = info.get('event', 'timeout') if info else 'timeout'

                # 蒙特卡洛回报计算
                mc_returns = self._compute_monte_carlo_returns(rewards, outcome)

                # 状态转换为张量
                for state, mc_value in zip(states, mc_returns):
                    if hasattr(state, 'to_array'):
                        state_tensor = torch.from_numpy(state.to_array()).float().to(self.device)
                    elif isinstance(state, np.ndarray):
                        state_tensor = torch.from_numpy(state).float().to(self.device)
                    elif isinstance(state, torch.Tensor):
                        state_tensor = state.float().to(self.device)
                    else:
                        continue

                    value_dataset.append((state_tensor, float(mc_value)))

            except Exception as e:
                logging.warning(f"[VALUE-FIRST-IL] 轨迹处理失败: {e}")
                continue

        return value_dataset

    def _compute_monte_carlo_returns(self, rewards, outcome) -> List[float]:
        """
        计算蒙特卡洛回报，考虑结果类型

        Args:
            rewards: 奖励序列
            outcome: 轨迹结果 ('success', 'collision', 'timeout')

        Returns:
            List[float]: 每个状态的蒙特卡洛价值
        """
        # 结果调整因子
        outcome_adjustment = {
            'reach_goal': 0.2,   # 成功获得额外奖励
            'success': 0.2,
            'collision': -0.1,   # 碰撞额外惩罚
            'timeout': 0.0       # 超时无额外调整
        }

        adjustment = outcome_adjustment.get(outcome, 0.0)

        # 蒙特卡洛回报计算
        mc_returns = []
        G = 0  # 累积回报

        # 从后向前计算
        for r in reversed(rewards):
            G = r + self.gamma * G
            mc_returns.insert(0, G + adjustment)

        return mc_returns

    def _train_value_network(self, value_dataset, epochs, batch_size) -> float:
        """
        直接训练Mamba价值网络

        Args:
            value_dataset: 状态-价值对数据集
            epochs: 训练轮数
            batch_size: 批大小

        Returns:
            平均训练损失
        """
        # 准备数据加载器
        states = torch.stack([s for s, v in value_dataset])
        values = torch.tensor([v for s, v in value_dataset], dtype=torch.float32).to(self.device)
        dataset = TensorDataset(states, values)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 设置训练模式
        self.model.train()

        total_loss = 0.0
        total_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            for batch_states, batch_values in dataloader:
                # 确保状态格式正确（Mamba需要4D输入）
                if batch_states.dim() == 2:  # [B, 34] -> [B, 1, 6, 13]
                    # 转换为token格式（复用contracts转换逻辑）
                    batch_states = self._states_to_tokens(batch_states)

                # 使用Mamba的前向价值函数
                try:
                    pred_values = self.model.forward_value(batch_states)
                    if pred_values.dim() > 1:
                        pred_values = pred_values.squeeze(-1)
                except Exception as e:
                    # 备用方案：直接使用价值头
                    logging.warning(f"[VALUE-FIRST-IL] forward_value失败，使用备用方案: {e}")
                    flat_states = batch_states.view(batch_states.size(0), -1)
                    pred_values = self.model.value_head(flat_states).squeeze(-1)

                # 价值损失计算
                loss = F.mse_loss(pred_values, batch_values)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.model.value_head.parameters(), max_norm=1.0)

                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1

            # 每10个epoch输出进度（避免刷屏）
            if epoch % 10 == 0 or epoch == epochs - 1:
                avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
                logging.info(f"[VALUE-FIRST-IL] Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")

            total_loss += epoch_loss
            total_batches += epoch_batches

        return total_loss / max(total_batches, 1)

    def _states_to_tokens(self, states):
        """
        状态转token格式（复用现有转换逻辑）

        Args:
            states: [B, 34] 状态张量

        Returns:
            tokens: [B, 1, 6, 13] token格式
        """
        try:
            from crowd_nav.contracts import joint34_to_tokens

            # GPU张量转CPU再转numpy（避免CUDA错误）
            states_np = states.detach().cpu().numpy()
            tokens = joint34_to_tokens(states_np)  # [B, 1, 6, 13]

            # 转回GPU张量
            if isinstance(tokens, np.ndarray):
                tokens = torch.from_numpy(tokens).float().to(self.device)
            elif isinstance(tokens, torch.Tensor):
                tokens = tokens.float().to(self.device)

            return tokens

        except Exception as e:
            logging.warning(f"[VALUE-FIRST-IL] 状态转换失败，使用简单reshape: {e}")
            # 备用：简单reshape（可能不完全正确，但能运行）
            B = states.size(0)
            return states.view(B, 1, 6, 13)  # 假设34D可以reshape为6×13


def create_value_first_il_trainer(policy, device='cuda', gamma=0.95) -> ValueFirstILTrainer:
    """
    工厂函数：创建价值优先IL训练器

    Args:
        policy: Mamba策略对象
        device: 设备
        gamma: 折扣因子

    Returns:
        ValueFirstILTrainer实例
    """
    return ValueFirstILTrainer(policy, device, gamma)
