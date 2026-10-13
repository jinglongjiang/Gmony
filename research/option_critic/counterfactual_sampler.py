"""Counterfactual Advantage & Risk Sampler for CF-OC

Evaluates decision-time counterfactual branches from exact simulation snapshots:
1. Continue Branch: Holds current option o_t for K steps (e.g., K=16 = 4.0s).
2. Reselect Branch: Executes counterfactual option reselection o^* ~ pi_Omega for K steps.
3. Computes ground-truth counterfactual advantage:
   Delta_CF = G_reselect - G_continue
4. Computes ground-truth short-horizon collision indicator:
   y_collision = 1.0 if collision occurred within K steps, else 0.0.
"""

import copy
import math
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
KDARL_CLEAN = REPO_ROOT / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(REPO_ROOT)] + [p for p in sys.path if p not in (str(REPO_ROOT), str(KDARL_CLEAN), '')]

from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY


def clone_pre_decision_snapshot(env, policy):
    """Deep clone simulation and policy state strictly before step decision."""
    for h in env.humans:
        if hasattr(h.policy, 'sim'):
            h.policy.sim = None
    env_clone = copy.deepcopy(env)
    policy_state = policy.clone_state()
    return env_clone, policy_state


class CounterfactualSampler:
    """Manages same-state branching simulations to harvest counterfactual supervision."""

    def __init__(self, k_steps: int = 16, gamma: float = 0.99):
        self.k_steps = k_steps
        self.gamma = gamma

    def run_sub_rollout(
        self,
        env_snapshot,
        policy,
        policy_state: Dict,
        fixed_option: Optional[int] = None,
        force_reselect: bool = False
    ) -> Tuple[float, bool]:
        """Roll out K steps from snapshot and return (discounted_return, had_collision)."""
        for h in env_snapshot.humans:
            if hasattr(h.policy, 'sim'):
                h.policy.sim = None
        env_branch = copy.deepcopy(env_snapshot)
        robot = env_branch.robot
        robot.set_policy(policy)
        robot.env = env_branch
        env_branch.set_robot(robot)

        # Restore policy snapshot state
        policy.restore_state(policy_state)

        if force_reselect:
            # Force termination: clear current option so policy reselects on first step
            policy.current_option = None
        elif fixed_option is not None:
            policy.current_option = fixed_option

        discounted_return = 0.0
        cur_discount = 1.0
        had_collision = False
        step = 0
        done = False

        while not done and step < self.k_steps:
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env_branch.humans])
            
            # If fixed_option is set, force it to persist during the K-step evaluation window
            if fixed_option is not None:
                policy.current_option = fixed_option
                
            action_xy, _, _ = policy.act(state, deterministic=True)
            obs, rew, term, trunc, info = env_branch.step(action_xy)
            done = term or trunc

            discounted_return += cur_discount * rew
            cur_discount *= self.gamma
            step += 1

            last_event = info.get('event', 'nothing') if info else 'nothing'
            if last_event == 'collision':
                had_collision = True
                break

        return float(discounted_return), bool(had_collision)

    def evaluate_branches(
        self,
        env,
        policy,
        curr_option: int
    ) -> Dict[str, float]:
        """Execute paired Continue vs Reselect rollouts from current pre-decision state.
        
        Returns:
            Dictionary with:
            - 'delta_cf': G_reselect - G_continue
            - 'g_continue': return of continuing option
            - 'g_reselect': return of reselecting option
            - 'coll_continue': collision occurrence in continue branch (1.0 or 0.0)
            - 'coll_reselect': collision occurrence in reselect branch (1.0 or 0.0)
        """
        env_snap, pol_snap = clone_pre_decision_snapshot(env, policy)

        # Branch 1: Continue current option
        g_continue, coll_cont = self.run_sub_rollout(
            env_snap, policy, pol_snap, fixed_option=curr_option, force_reselect=False
        )

        # Branch 2: Reselect option
        g_reselect, coll_res = self.run_sub_rollout(
            env_snap, policy, pol_snap, fixed_option=None, force_reselect=True
        )

        # Restore original state back onto the environment & policy
        policy.restore_state(pol_snap)

        delta_cf = g_reselect - g_continue
        return {
            'delta_cf': float(delta_cf),
            'g_continue': float(g_continue),
            'g_reselect': float(g_reselect),
            'coll_continue': 1.0 if coll_cont else 0.0,
            'coll_reselect': 1.0 if coll_res else 0.0
        }
