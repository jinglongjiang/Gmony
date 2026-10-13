# -*- coding: utf-8 -*-
"""
Replay Buffer - Off-Policy Data Collection (Supports DoubleQ Discrete Actions)
"""

import numpy as np
import torch
from collections import deque
import logging


# ========== Replay Buffer (Off-Policy, 支持离散动作) ==========

class ReplayBufferIQL:
    """Universal Replay Buffer (Off-Policy)

    Features:
    - Stores transitions (s, a, r, s', done)
    - Supports discrete action indices (action_indices) for DoubleQ
    - Retains BC/IL data long-term (no clearing)
    - Supports random sampling
    - Can mix IL+RL data
    - Stores Monte Carlo Returns (G_t) for stable regression target
    """
    def __init__(self, capacity=200000, seq_len=12, obs_shape=(8, 13),
                 n_step: int = 1, gamma: float = 0.99,
                 use_per: bool = False, per_alpha: float = 0.6, per_beta: float = 0.4, per_eps: float = 1e-6,
                 history_contract: str = 'legacy'):
        self.capacity = capacity
        self.seq_len = seq_len
        self.obs_shape = obs_shape
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.use_per = bool(use_per)
        self.per_alpha = float(per_alpha)
        self.per_beta = float(per_beta)
        self.per_eps = float(per_eps)
        self.history_contract = history_contract
        if history_contract not in ('legacy', 'legal-prefix'):
            raise ValueError('Unknown replay history contract')

        # Circular buffer for transitions
        self.ptr = 0
        self.size = 0

        # Pre-allocate arrays
        self.states = np.zeros((capacity, seq_len, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, 2), dtype=np.float32)
        self.action_indices = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, seq_len, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        # Monte Carlo Returns
        self.returns = np.zeros(capacity, dtype=np.float32)
        
        # PER
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

        logging.info(f"[IQL-REPLAY] Initialized: capacity={capacity}, seq_len={seq_len}, with MC returns")

    def store(self, state, action, reward, next_state, done, action_idx=None, priority: float = None, mc_return: float = 0.0):
        """Store a single transition

        Args:
            state: [T, 8, 13] tokens
            action: [2] continuous action (kept for debug)
            reward: scalar
            next_state: [T, 8, 13] tokens
            done: bool
            action_idx: int, discrete action index [0, 79]
            mc_return: float, full trajectory return from this step
        """
        # Ensure state is [seq_len, 8, 13]
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.detach().cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)

        # Truncate or pad to seq_len
        if state.shape[0] > self.seq_len:
            state = state[-self.seq_len:]
        elif state.shape[0] < self.seq_len:
            pad_len = self.seq_len - state.shape[0]
            pad = np.zeros_like(state[:1]) if self.history_contract == 'legal-prefix' else state[:1]
            state = np.concatenate([np.repeat(pad, pad_len, axis=0), state], axis=0)

        if next_state.shape[0] > self.seq_len:
            next_state = next_state[-self.seq_len:]
        elif next_state.shape[0] < self.seq_len:
            pad_len = self.seq_len - next_state.shape[0]
            pad = np.zeros_like(next_state[:1]) if self.history_contract == 'legal-prefix' else next_state[:1]
            next_state = np.concatenate([np.repeat(pad, pad_len, axis=0), next_state], axis=0)

        self.states[self.ptr] = state
        self.actions[self.ptr] = action[:2]  # Only take [vx, vy]
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.returns[self.ptr] = float(mc_return)

        # Store discrete action index
        if action_idx is not None:
            self.action_indices[self.ptr] = action_idx
        # PER priority
        if self.use_per:
            if priority is None:
                priority = self.max_priority
            priority = float(priority)
            self.priorities[self.ptr] = priority
            if priority > self.max_priority:
                self.max_priority = priority

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def store_episode(self, episode_data):
        """Split episode data into transitions and store

        Args:
            episode_data: dict with keys:
                - states: [T, 34] or tokens [T, 6, 13]
                - actions_continuous: [T, 2]
                - rewards: [T]
                - dones: [T]
        """
        from crowd_nav.contracts import joint34_to_tokens

        states_raw = episode_data['states']
        actions = episode_data.get('actions_continuous', episode_data.get('actions', []))
        rewards = episode_data['rewards']
        dones = episode_data.get('dones', np.zeros(len(states_raw), dtype=bool))

        T = len(states_raw)
        if T <= 1:
            return  # Need at least two steps to form a transition

        # Convert states to tokens
        if 'tokens' in episode_data and episode_data['tokens'] is not None:
            states_tokens = episode_data['tokens']
        else:
            states_tokens = joint34_to_tokens(np.array(states_raw, dtype=np.float32))

        if isinstance(states_tokens, torch.Tensor):
            states_tokens = states_tokens.detach().cpu().numpy()
        states_tokens = np.asarray(states_tokens, dtype=np.float32)

        # Ensure [T, 6, 13]
        if states_tokens.ndim == 4 and states_tokens.shape[1] == 1:
            states_tokens = np.squeeze(states_tokens, axis=1)

        # Only use externally provided discrete action indices (forbid reverse engineering from continuous)
        actions_arr = np.array(actions, dtype=np.float32)[:, :2]
        action_indices_arr = episode_data.get('action_indices', None)
        if action_indices_arr is None:
            logging.warning("[REPLAY] Missing action_indices; skip store_episode to avoid mismatched action grid.")
            return
        if any(a is None for a in action_indices_arr):
            logging.warning("[REPLAY] action_indices contains None; skip store_episode.")
            return
        action_indices_arr = np.array(action_indices_arr, dtype=np.int64)
        if action_indices_arr.shape[0] != T:
            logging.warning(f"[REPLAY] action_indices length mismatch: {action_indices_arr.shape[0]} vs T={T}. Skip store.")
            return

        # Pre-calculate Monte Carlo returns (G_t)
        # G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} ...
        mc_returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            # SARL uses 0.99 gamma usually
            G = rewards[t] + self.gamma * G
            mc_returns[t] = G

        # Split into transitions. Keep the final transition: it carries the
        # terminal success/collision/timeout reward used by discrete Q-learning.
        for t in range(T):
            n = self.n_step
            end = min(t + n, T - 1)
            # n-step return (still kept for reference, but MC return is preferred for SARL regression)
            Rn = 0.0
            done_n = False
            for k in range(n):
                idx = t + k
                if idx >= T:
                    break
                Rn += (self.gamma ** k) * float(rewards[idx])
                if bool(dones[idx]):
                    done_n = True
                    end = min(idx + 1, T - 1)
                    break

            # Build history window
            if t + 1 >= self.seq_len:
                state_t = states_tokens[t + 1 - self.seq_len:t + 1]
            else:
                pad_len = self.seq_len - (t + 1)
                state_t = np.concatenate([
                    np.repeat(np.zeros_like(states_tokens[:1]) if self.history_contract == 'legal-prefix'
                              else states_tokens[:1], pad_len, axis=0),
                    states_tokens[:t+1]
                ], axis=0)

            # Window for s_{t+n}
            if end + 1 >= self.seq_len:
                next_state_t = states_tokens[end + 1 - self.seq_len:end + 1]
            else:
                pad_len = self.seq_len - (end + 1)
                next_state_t = np.concatenate([
                    np.repeat(np.zeros_like(states_tokens[:1]) if self.history_contract == 'legal-prefix'
                              else states_tokens[:1], pad_len, axis=0),
                    states_tokens[:end+1]
                ], axis=0)

            # Store with action_idx and MC return
            self.store(
                state=state_t,
                action=actions_arr[t],
                reward=Rn,
                next_state=next_state_t,
                done=done_n,
                action_idx=int(action_indices_arr[t]),
                mc_return=mc_returns[t]
            )

    def push_episode(self, *args, **kwargs):
        """兼容Explorer接口的别名方法"""
        return self.store_episode(*args, **kwargs)

    def sample(self, batch_size, device='cpu'):
        """随机采样batch

        Returns:
            dict with keys:
                - states: [B, T, 6, 13]
                - actions: [B, 2]
                - rewards: [B]
                - next_states: [B, T, 6, 13]
                - dones: [B]
                - returns: [B] (MC Return)
        """
        if self.use_per:
            prios = self.priorities[:self.size] + self.per_eps
            probs = prios ** self.per_alpha
            probs = probs / probs.sum()
            indices = np.random.choice(self.size, batch_size, p=probs)
            weights = (self.size * probs[indices]) ** (-self.per_beta)
            weights = weights / weights.max()
            weights_t = torch.tensor(weights, dtype=torch.float32, device=device).view(-1, 1)
        else:
            indices = np.random.randint(0, self.size, size=batch_size)
            weights_t = None

        batch = {
            'states': torch.tensor(self.states[indices], dtype=torch.float32, device=device),
            'actions': torch.tensor(self.actions[indices], dtype=torch.float32, device=device),
            'action_indices': torch.tensor(self.action_indices[indices], dtype=torch.long, device=device),
            'rewards': torch.tensor(self.rewards[indices], dtype=torch.float32, device=device),
            'next_states': torch.tensor(self.next_states[indices], dtype=torch.float32, device=device),
            'dones': torch.tensor(self.dones[indices], dtype=torch.float32, device=device),
            'returns': torch.tensor(self.returns[indices], dtype=torch.float32, device=device),  # ✅ 新增：MC Return
            'indices': indices
        }
        if weights_t is not None:
            batch['weights'] = weights_t

        return batch

    def update_priorities(self, indices, priorities):
        if not self.use_per:
            return
        if indices is None:
            return
        for idx, prio in zip(indices, priorities):
            p = float(prio)
            if p <= 0:
                p = self.per_eps
            self.priorities[int(idx)] = p
            if p > self.max_priority:
                self.max_priority = p

    def set_per_beta(self, beta: float):
        self.per_beta = float(beta)

    def __len__(self):
        return self.size

    def clear(self):
        """清空buffer（IQL通常不需要，但提供接口）"""
        self.ptr = 0
        self.size = 0


class ReplayBufferMC(ReplayBufferIQL):
    """MC-only replay: 去掉纯 Monte-Carlo 价值回归从不消费的字段与窗口。

    源码依据 (train.py sarl_style_update): 只读 batch['states'] 与 batch['returns'];
    next_states / rewards / dones / actions 在 MC 路径上从不被读取 (TD bootstrap 已注释掉)。
    mc_only=False 时行为与父类完全一致, 不影响 DoubleQ 路径。
    """

    def __init__(self, *args, mc_only=True, **kwargs):
        self.mc_only = bool(mc_only)
        super().__init__(*args, **kwargs)
        if self.mc_only:
            freed = self.next_states.nbytes
            self.next_states = None
            logging.info(f"[MC-REPLAY] next_states 未分配, 省 {freed / 2 ** 30:.2f} GiB")

    def store(self, state, action, reward, next_state, done,
              action_idx=None, priority=None, mc_return=0.0):
        if not self.mc_only:
            return super().store(state, action, reward, next_state, done,
                                 action_idx, priority, mc_return)
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        state = np.asarray(state, dtype=np.float32)
        if state.shape[0] > self.seq_len:
            state = state[-self.seq_len:]
        elif state.shape[0] < self.seq_len:
            pad_len = self.seq_len - state.shape[0]
            state = np.concatenate([np.repeat((np.zeros_like(state[:1]) if self.history_contract == 'legal-prefix' else state[:1]), pad_len, axis=0), state], axis=0)
        p = self.ptr
        self.states[p] = state
        self.actions[p] = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        self.action_indices[p] = int(action_idx) if action_idx is not None else 0
        self.rewards[p] = float(reward)
        self.dones[p] = float(bool(done))
        self.returns[p] = float(mc_return)
        if self.use_per:
            priority = self.max_priority if priority is None else float(priority)
            self.priorities[p] = priority
            self.max_priority = max(self.max_priority, priority)
        self.ptr = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def store_episode(self, episode_data):
        if not self.mc_only:
            return super().store_episode(episode_data)
        from crowd_nav.contracts import joint34_to_tokens
        states_raw = episode_data['states']
        actions = episode_data.get('actions_continuous', episode_data.get('actions', []))
        rewards = episode_data['rewards']
        dones = episode_data.get('dones', np.zeros(len(states_raw), dtype=bool))
        T = len(states_raw)
        if T <= 1:
            return
        if 'tokens' in episode_data and episode_data['tokens'] is not None:
            tok = episode_data['tokens']
        else:
            tok = joint34_to_tokens(np.array(states_raw, dtype=np.float32))
        if isinstance(tok, torch.Tensor):
            tok = tok.detach().cpu().numpy()
        tok = np.asarray(tok, dtype=np.float32)
        if tok.ndim == 4 and tok.shape[1] == 1:
            tok = np.squeeze(tok, axis=1)
        actions_arr = np.array(actions, dtype=np.float32)[:, :2]
        ai = episode_data.get('action_indices', None)
        if ai is None:
            logging.warning("[REPLAY] Missing action_indices; skip store_episode.")
            return
        if any(a is None for a in ai):
            logging.warning("[REPLAY] action_indices contains None; skip store_episode.")
            return
        ai = np.array(ai, dtype=np.int64)
        if ai.shape[0] != T:
            logging.warning(f"[REPLAY] action_indices length mismatch: {ai.shape[0]} vs T={T}. Skip store.")
            return
        mc = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            mc[t] = G
        L = self.seq_len
        for t in range(T):
            n = self.n_step
            Rn = 0.0
            done_n = False
            for k in range(n):
                idx = t + k
                if idx >= T:
                    break
                Rn += (self.gamma ** k) * float(rewards[idx])
                if bool(dones[idx]):
                    done_n = True
                    break
            if t + 1 >= L:
                state_t = tok[t + 1 - L:t + 1]
            else:
                pad_len = L - (t + 1)
                state_t = np.concatenate([
                    np.repeat((np.zeros_like(tok[:1]) if self.history_contract == 'legal-prefix' else tok[:1]), pad_len, axis=0), tok[:t + 1]], axis=0)
            self.store(state=state_t, action=actions_arr[t], reward=Rn,
                       next_state=None, done=done_n,
                       action_idx=int(ai[t]), mc_return=mc[t])

    def sample(self, batch_size, device='cpu'):
        if not self.mc_only:
            return super().sample(batch_size, device)
        if self.use_per:
            prios = self.priorities[:self.size] + self.per_eps
            probs = prios ** self.per_alpha
            probs = probs / probs.sum()
            indices = np.random.choice(self.size, batch_size, p=probs)
            w = (self.size * probs[indices]) ** (-self.per_beta)
            w = w / w.max()
            weights_t = torch.tensor(w, dtype=torch.float32, device=device).view(-1, 1)
        else:
            indices = np.random.randint(0, self.size, size=batch_size)
            weights_t = None
        return {'states': torch.tensor(self.states[indices], dtype=torch.float32, device=device),
                'returns': torch.tensor(self.returns[indices], dtype=torch.float32, device=device),
                'indices': indices, 'weights': weights_t}


class ReplayBufferSAC(ReplayBufferIQL):
    """
    SAC专用Replay Buffer（与IQL实现一致，但独立命名和日志，避免SAC路径出现IQL标签）
    """
    def __init__(self, capacity=200000, seq_len=12, obs_shape=(8, 13)):
        super().__init__(capacity=capacity, seq_len=seq_len, obs_shape=obs_shape)
        logging.info(f"[SAC-REPLAY] Initialized: capacity={capacity}, seq_len={seq_len}")


class ReplayBufferDoubleQ(ReplayBufferIQL):
    """
    DoubleQ专用Replay Buffer（与IQL实现一致，但独立命名和日志）
    """
    def __init__(self, capacity=200000, seq_len=12, obs_shape=(8, 13)):
        super().__init__(capacity=capacity, seq_len=seq_len, obs_shape=obs_shape)
        logging.info(f"[DOUBLEQ-REPLAY] Initialized: capacity={capacity}, seq_len={seq_len}")
