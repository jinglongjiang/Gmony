"""Hierarchical Replay Buffer for Option-Critic & CF-OC Training

Stores trajectory transitions and provides:
1. Multi-step option transition indexing: (obs, option, action, reward, next_obs, done).
2. Backward propagation of short-horizon collision labels (16 steps = 4s).
3. Counterfactual advantage assignment (Delta_CF = G_reselect - G_continue).
4. GPU tensor batch sampling with pinned memory and action masking.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


class OptionReplayBuffer:
    """Experience replay buffer tailored for hierarchical Option-Critic architectures."""

    def __init__(
        self,
        capacity: int = 50000,
        seq_len: int = 24,
        num_actions: int = 80,
        collision_horizon: int = 16,
        device: str = 'cpu'
    ):
        self.capacity = capacity
        self.seq_len = seq_len
        self.num_actions = num_actions
        self.collision_horizon = collision_horizon
        self.device = torch.device(device) if isinstance(device, str) else device

        # Preallocated numpy memory
        self.obs_buf = np.zeros((capacity, seq_len, 8, 13), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, seq_len, 8, 13), dtype=np.float32)
        self.option_buf = np.zeros((capacity,), dtype=np.int64)
        self.action_buf = np.zeros((capacity,), dtype=np.int64)
        self.reward_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)
        self.mask_buf = np.ones((capacity, num_actions), dtype=np.float32)

        # CF-OC specific supervisory labels
        self.cf_adv_buf = np.zeros((capacity,), dtype=np.float32)
        self.has_cf_buf = np.zeros((capacity,), dtype=np.float32)
        self.coll_risk_buf = np.zeros((capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

        # Episode staging buffer
        self._curr_episode_indices: List[int] = []

    def add(
        self,
        obs: np.ndarray,
        option: int,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        action_mask: Optional[np.ndarray] = None,
        cf_adv: Optional[float] = None
    ) -> int:
        """Store a single step transition into the active episode buffer."""
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.next_obs_buf[idx] = next_obs
        self.option_buf[idx] = option
        self.action_buf[idx] = action
        self.reward_buf[idx] = reward
        self.done_buf[idx] = float(done)

        if action_mask is not None:
            self.mask_buf[idx] = action_mask
        else:
            self.mask_buf[idx] = 1.0

        if cf_adv is not None:
            self.cf_adv_buf[idx] = float(cf_adv)
            self.has_cf_buf[idx] = 1.0
        else:
            self.cf_adv_buf[idx] = 0.0
            self.has_cf_buf[idx] = 0.0

        # Collision risk default (overwritten on episode finish)
        self.coll_risk_buf[idx] = 0.0

        self._curr_episode_indices.append(idx)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def finish_episode(self, end_event: str):
        """Finalize episode: tag short-horizon collision risk if ended in collision."""
        is_collision = (str(end_event).lower() == 'collision')
        if is_collision and len(self._curr_episode_indices) > 0:
            # Tag the last collision_horizon steps leading to the collision
            coll_steps = self._curr_episode_indices[-self.collision_horizon:]
            for idx in coll_steps:
                self.coll_risk_buf[idx] = 1.0

        self._curr_episode_indices.clear()

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions uniformly at random."""
        if self.size < batch_size:
            raise ValueError(f"Buffer has {self.size} samples, fewer than batch_size {batch_size}")

        indices = np.random.randint(0, self.size, size=batch_size)

        batch = {
            'obs': torch.from_numpy(self.obs_buf[indices]).float().to(self.device),
            'next_obs': torch.from_numpy(self.next_obs_buf[indices]).float().to(self.device),
            'option': torch.from_numpy(self.option_buf[indices]).long().to(self.device),
            'action': torch.from_numpy(self.action_buf[indices]).long().to(self.device),
            'reward': torch.from_numpy(self.reward_buf[indices]).float().to(self.device),
            'done': torch.from_numpy(self.done_buf[indices]).float().to(self.device),
            'action_mask': torch.from_numpy(self.mask_buf[indices]).float().to(self.device),
            'cf_adv': torch.from_numpy(self.cf_adv_buf[indices]).float().to(self.device),
            'has_cf': torch.from_numpy(self.has_cf_buf[indices]).float().to(self.device),
            'coll_risk': torch.from_numpy(self.coll_risk_buf[indices]).float().to(self.device)
        }
        return batch

    def __len__(self) -> int:
        return self.size
