"""OptionCriticPolicy: Modular Execution Wrapper for CrowdSim

Integrates:
1. Call-and-Return Option Execution Cycle (Bacon et al. AAAI 2017).
2. Intra-option action sampling with safety masking.
3. Support for Standard OC, Deliberation Cost OC, and CF-OC.
4. Comprehensive behavioral telemetry:
   - Option dwell duration tracking
   - Termination probability dynamics
   - Switching rate
   - Counterfactual advantage calibration logs
   - Short-horizon collision risk predictions
5. Full cloning compatibility for counterfactual tree branching.
"""

from collections import deque
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
KDARL_CLEAN = REPO_ROOT / 'kdarl_clean'
if str(KDARL_CLEAN) not in sys.path:
    sys.path.insert(0, str(KDARL_CLEAN))

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import JointState
from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized, discrete_index_to_action
from research.option_critic.networks import OptionCriticNetwork


class OptionCriticPolicy(nn.Module):
    """Hierarchical Option-Critic Policy for crowd navigation."""

    def __init__(
        self,
        config=None,
        network: Optional[OptionCriticNetwork] = None,
        device: str = 'cpu',
        num_options: int = 4,
        num_actions: int = 80,
        seq_len: int = 24,
        is_cf_oc: bool = True,
        deliberation_cost: float = 0.05,
        termination_threshold: float = 0.5,
        use_sampling_termination: bool = False
    ):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.num_options = num_options
        self.num_actions = num_actions
        self.seq_len = seq_len
        self.is_cf_oc = is_cf_oc
        self.deliberation_cost = deliberation_cost
        self.termination_threshold = termination_threshold
        self.use_sampling_termination = use_sampling_termination

        # Config fallback
        if config is not None:
            if hasattr(config, 'getfloat'):
                self.time_step = config.getfloat('env', 'time_step', fallback=0.25)
            else:
                self.time_step = 0.25
        else:
            self.time_step = 0.25

        # Build network if not provided
        if network is not None:
            self.net = network
        else:
            self.net = OptionCriticNetwork(
                num_options=num_options,
                num_actions=num_actions,
                d_model=256,
                n_layers=4,
                temporal_backbone='gru',
                is_cf_oc=is_cf_oc
            )
        self.net.to(self.device)

        # Execution state
        self.current_option: Optional[int] = None
        self.option_duration: int = 0
        self.total_switches: int = 0
        self.step_count: int = 0
        self.epsilon: float = 0.0

        # State history buffer: [seq_len, 8, 13]
        self._history = deque(maxlen=self.seq_len)

        # Telemetry tracking
        self.telemetry = {
            'option_history': [],
            'termination_probs': [],
            'cf_advantages': [],
            'collision_risks': [],
            'switch_events': [],
            'durations': []
        }

    def reset_episode_stats(self):
        """Reset history and internal option state at start of episode."""
        self._history.clear()
        self.current_option = None
        self.option_duration = 0
        self.total_switches = 0
        self.step_count = 0
        self.telemetry = {
            'option_history': [],
            'termination_probs': [],
            'cf_advantages': [],
            'collision_risks': [],
            'switch_events': [],
            'durations': []
        }

    def _state_to_tokens(self, state) -> np.ndarray:
        """Convert environment state to 8x13 token format."""
        if hasattr(state, 'to_array'):
            state_34 = state.to_array()
        elif isinstance(state, (list, tuple)):
            if len(state) > 0 and hasattr(state[0], 'to_array'):
                state_34 = np.concatenate([s.to_array() if hasattr(s, 'to_array') else np.array(s, dtype=np.float32) for s in state])
            else:
                state_34 = np.array(state, dtype=np.float32).flatten()
        else:
            state_34 = np.array(state, dtype=np.float32).flatten()

        state_34 = state_34.flatten()
        if state_34.size < 34:
            pad = np.zeros(34 - state_34.size, dtype=np.float32)
            state_34 = np.concatenate([state_34, pad])
        elif state_34.size > 34:
            state_34 = state_34[:34]

        return _batch_joint34_to_tokens_vectorized(state_34.reshape(1, -1))[0]

    def _prepare_history_tensor(self, current_token: np.ndarray) -> torch.Tensor:
        """Append token to history and return [1, seq_len, 8, 13] tensor."""
        self._history.append(current_token.copy())
        hist_list = list(self._history)
        if len(hist_list) < self.seq_len:
            pad_frame = hist_list[0]
            pad_needed = self.seq_len - len(hist_list)
            hist_list = [pad_frame.copy() for _ in range(pad_needed)] + hist_list

        hist_list = hist_list[-self.seq_len:]
        hist_np = np.array(hist_list, dtype=np.float32)
        return torch.from_numpy(hist_np).float().to(self.device).unsqueeze(0)

    def select_option(
        self,
        q_omega: torch.Tensor,
        deterministic: bool = True
    ) -> int:
        """Select an option based on option values Q_Omega(h, o)."""
        if not deterministic and self.epsilon > 0 and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.num_options))
        return int(q_omega.argmax(dim=-1).item())

    def act(
        self,
        state,
        deterministic: bool = True,
        action_mask: Optional[np.ndarray] = None
    ) -> Tuple[ActionXY, int, int]:
        """Execute one step of the Call-and-Return Option-Critic loop.
        
        Returns:
            action_xy: Continuous ActionXY
            action_idx: Discrete action index in [0, 79]
            option_idx: Selected option index in [0, num_options - 1]
        """
        token = self._state_to_tokens(state)
        hist_tensor = self._prepare_history_tensor(token)  # [1, 24, 8, 13]

        mask_tensor = None
        if action_mask is not None:
            mask_tensor = torch.from_numpy(action_mask).float().to(self.device).unsqueeze(0)

        self.net.eval()
        with torch.no_grad():
            out = self.net(hist_tensor, action_mask=mask_tensor)
            q_omega = out['q_omega']  # [1, num_options]
            beta_probs = out['beta_cf'] if (self.is_cf_oc and 'beta_cf' in out) else out['beta']  # [1, num_options]
            policy_probs = out['policy_probs']  # [1, num_options, num_actions]

            # 1. Call-and-Return Termination & Option Selection
            is_switched = False
            if self.current_option is None:
                # Initial step: choose option
                self.current_option = self.select_option(q_omega, deterministic=deterministic)
                self.option_duration = 0
            else:
                # Step t > 0: evaluate termination of current option
                term_prob = float(beta_probs[0, self.current_option].item())
                should_terminate = False
                
                if self.use_sampling_termination and not deterministic:
                    should_terminate = (np.random.random() < term_prob)
                else:
                    should_terminate = (term_prob >= self.termination_threshold)

                if should_terminate:
                    old_opt = self.current_option
                    new_opt = self.select_option(q_omega, deterministic=deterministic)
                    if new_opt != old_opt:
                        is_switched = True
                        self.total_switches += 1
                        self.telemetry['durations'].append(self.option_duration)
                    self.current_option = new_opt
                    self.option_duration = 0
                else:
                    self.option_duration += 1

            # 2. Intra-Option Action Selection
            curr_opt = self.current_option
            opt_probs = policy_probs[0, curr_opt]  # [num_actions]

            if not deterministic and self.epsilon > 0 and np.random.random() < self.epsilon:
                if action_mask is not None:
                    valid_indices = np.where(action_mask > 0.5)[0]
                    action_idx = int(np.random.choice(valid_indices)) if len(valid_indices) > 0 else 0
                else:
                    action_idx = int(np.random.randint(0, self.num_actions))
            elif deterministic:
                action_idx = int(opt_probs.argmax().item())
            else:
                probs_np = opt_probs.cpu().numpy()
                probs_np = np.nan_to_num(probs_np, nan=0.0)
                sum_p = probs_np.sum()
                if sum_p > 1e-6:
                    probs_np /= sum_p
                    action_idx = int(np.random.choice(self.num_actions, p=probs_np))
                else:
                    action_idx = int(opt_probs.argmax().item())

            vx, vy = discrete_index_to_action(action_idx)

            # Record telemetry
            self.telemetry['option_history'].append(curr_opt)
            self.telemetry['termination_probs'].append(float(beta_probs[0, curr_opt].item()))
            self.telemetry['switch_events'].append(1 if is_switched else 0)
            if self.is_cf_oc and 'cf_advantage' in out:
                self.telemetry['cf_advantages'].append(float(out['cf_advantage'][0, curr_opt].item()))
            if self.is_cf_oc and 'collision_risk' in out:
                self.telemetry['collision_risks'].append(float(out['collision_risk'][0, curr_opt].item()))

            self.step_count += 1
            return ActionXY(vx, vy), action_idx, curr_opt

    def predict(self, state, deterministic: bool = True, return_idx: bool = False, **kwargs):
        """Unified compatibility interface for CrowdSim benchmark runner."""
        action_xy, action_idx, option_idx = self.act(state, deterministic=deterministic)
        if return_idx:
            return action_xy, action_idx
        return action_xy

    def clone_state(self) -> Dict:
        """Deep copy internal policy execution state for counterfactual branching."""
        return {
            'current_option': self.current_option,
            'option_duration': self.option_duration,
            'total_switches': self.total_switches,
            'step_count': self.step_count,
            'history': [f.copy() for f in self._history]
        }

    def restore_state(self, state_dict: Dict):
        """Restore internal policy execution state."""
        self.current_option = state_dict['current_option']
        self.option_duration = state_dict['option_duration']
        self.total_switches = state_dict['total_switches']
        self.step_count = state_dict['step_count']
        self._history.clear()
        for f in state_dict['history']:
            self._history.append(f.copy())
