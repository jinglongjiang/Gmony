import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import sys
import math
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)

REPO_ROOT = Path('/home/abc/workspace/autonomous_research_20261012')
sys.path.insert(0, str(REPO_ROOT))

from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from run_b0_counterfactual_study import get_base_config
from run_selection_study_50cases import install_selection
from run_stage4_500cases_evaluation import (
    Stage4ControllerPolicy,
    load_stage4_policy
)

class ActionLevelControllerPolicy(Stage4ControllerPolicy):
    def __init__(
        self,
        mode='base',  # 'base', 'intent', 'action_hysteresis', 'action_repeat'
        config=None,
        device='cpu',
        alpha_a=0.10,
        k_repeat=2
    ):
        # If mode is 'intent', use Stage4 heuristic gate (K=4, alpha=0.10)
        super().__init__(mode='heuristic' if mode == 'intent' else 'base', config=config, device=device)
        self.action_mode = mode
        self.alpha_a = float(alpha_a)
        self.k_repeat = int(k_repeat)

        # Action-level commitment tracking buffers
        self.last_executed_cmd = None
        self.repeat_steps_remaining = 0
        self.locked_action_idx = None
        self.history_grid_indices = []
        self.history_actions = []
        self.grid_switch_count = 0
        self.vel_switch_count = 0

    def reset_episode_stats(self):
        super().reset_episode_stats()
        self.last_executed_cmd = None
        self.repeat_steps_remaining = 0
        self.locked_action_idx = None
        self.history_grid_indices.clear()
        self.history_actions.clear()
        self.grid_switch_count = 0
        self.vel_switch_count = 0

    def score_sarl_candidates(self, state: JointState):
        if self.action_mode == 'base':
            return super(Stage4ControllerPolicy, self).score_sarl_candidates(state)
        elif self.action_mode == 'intent':
            return super().score_sarl_candidates(state)
        elif self.action_mode == 'action_hysteresis':
            base_res = super(Stage4ControllerPolicy, self).score_sarl_candidates(state)
            commands = base_res['commands']
            scores = base_res['scores']

            if self.last_executed_cmd is not None:
                # Find candidate command with minimum Euclidean distance to previous executed command
                prev_vx = self.last_executed_cmd.vx
                prev_vy = self.last_executed_cmd.vy
                dists = [(c.vx - prev_vx)**2 + (c.vy - prev_vy)**2 for c in commands]
                closest_idx = int(np.argmin(dists))
                scores[closest_idx] += self.alpha_a

                # Re-apply safety filter if test_min_clearance > 0
                if self._phase in ('test', 'val', 'eval') and self.test_min_clearance > 0.0:
                    safe_mask = base_res['clearance'] >= self.test_min_clearance
                    if safe_mask.any():
                        scores = scores.masked_fill(~safe_mask, -1e9)

            base_res['scores'] = scores
            return base_res
        elif self.action_mode == 'action_repeat':
            base_res = super(Stage4ControllerPolicy, self).score_sarl_candidates(state)
            scores = base_res['scores']

            if self.repeat_steps_remaining > 0 and self.locked_action_idx is not None:
                # Repeat phase: force selection of locked action
                scores[self.locked_action_idx] += 1e6
                if self._phase in ('test', 'val', 'eval') and self.test_min_clearance > 0.0:
                    safe_mask = base_res['clearance'] >= self.test_min_clearance
                    if safe_mask.any():
                        scores = scores.masked_fill(~safe_mask, -1e9)
                self.repeat_steps_remaining -= 1
            else:
                # Fresh decision step
                chosen_idx = scores.argmax().item()
                self.locked_action_idx = chosen_idx
                self.repeat_steps_remaining = self.k_repeat - 1

            base_res['scores'] = scores
            return base_res
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

    def predict(self, state):
        act = super().predict(state)
        self.last_executed_cmd = act

        grid_idx = getattr(self, '_last_grid_index', None)
        if len(self.history_grid_indices) > 0 and grid_idx is not None:
            if grid_idx != self.history_grid_indices[-1]:
                self.grid_switch_count += 1

        if len(self.history_actions) > 0:
            prev_vx, prev_vy = self.history_actions[-1]
            if (act.vx - prev_vx)**2 + (act.vy - prev_vy)**2 > 1e-4:
                self.vel_switch_count += 1

        self.history_grid_indices.append(grid_idx)
        self.history_actions.append((act.vx, act.vy))
        return act

    def get_switching_rates(self):
        steps = len(self.history_grid_indices)
        denom = max(1, steps - 1)
        return {
            'grid_switch_count': self.grid_switch_count,
            'vel_switch_count': self.vel_switch_count,
            'total_steps': steps,
            'grid_switch_rate': round(self.grid_switch_count / denom, 4),
            'vel_switch_rate': round(self.vel_switch_count / denom, 4)
        }

def load_action_level_policy(mode='base', alpha_a=0.10, k_repeat=2, device='cpu'):
    cfg = get_base_config()
    policy = ActionLevelControllerPolicy(
        mode=mode,
        config=cfg,
        device=device,
        alpha_a=alpha_a,
        k_repeat=k_repeat
    )
    ckpt_path = REPO_ROOT / 'baseline_weights/gru-t24-seed443.pth'
    ckpt = torch.load(ckpt_path, map_location=device)
    raw_state = ckpt.get('policy_state', ckpt.get('model', ckpt))
    clean_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in raw_state.items()}
    policy.load_state_dict(clean_dict, strict=True)
    policy.eval()
    policy.build_action_space(1.0)
    policy.use_sarl_predict = True
    policy.set_phase('test')
    policy.test_min_clearance = 0.2
    policy.test_risk_lambda = 0.8
    policy.test_action_smoothing = 0.3
    install_selection(policy, 'ttc5')
    return policy
