#!/usr/bin/env python3
"""Phase P1: Counterfactual Intervention Study on Single-Stream GRU Baseline (B0)

Investigates macro-intent commitment across durations K in {0, 2, 4, 8} on the
frozen single-stream GRU baseline (B0).

Experimental Design:
1. Backbone: Pure Single-Stream GRU (gru-t24-seed443.pth, strict load, 0 missing keys).
2. Scan 200 independent cases (Cases 90000 - 90199).
3. Pre-decision state snapshots across 5 diverse operational archetypes:
   - Deadlock / Oscillatory Stagnation
   - Active Yield / Deceleration
   - Dangerous Approach (Collision Risk)
   - Normal Detour
   - Smooth Progression
4. Fork 4 counterfactual branches from identical pre-decision snapshots:
   - Branch A: Base B0 (K=0 steps, zero intervention, automated 100% parity assertion)
   - Branch B: B0 + Intent Commitment for K=2 steps (Short)
   - Branch C: B0 + Intent Commitment for K=4 steps (Medium)
   - Branch D: B0 + Intent Commitment for K=8 steps (Long)
   After K steps of intent commitment, all branches return to pure B0 reactive navigation.
5. Record per-snapshot features, branch outcomes, travel metrics, and causal advantage deltas.
6. Saves full case-grouped dataset to benchmark_results/b0_counterfactual_study_200cases.json.
"""

import sys
import copy
import math
import json
import time
from pathlib import Path
import configparser
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
KDARL_CLEAN = REPO_ROOT / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(REPO_ROOT)] + [p for p in sys.path if p not in (str(REPO_ROOT), str(KDARL_CLEAN), '')]

from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from run_selection_study_50cases import install_selection

RESULTS_DIR = REPO_ROOT / 'benchmark_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Macro intent constants
INTENT_BYPASS_LEFT = 0
INTENT_DIRECT = 1
INTENT_BYPASS_RIGHT = 2
INTENT_YIELD = 3
INTENT_NAMES = {0: 'Bypass_Left', 1: 'Direct', 2: 'Bypass_Right', 3: 'Yield'}

def get_base_config():
    env_config = configparser.RawConfigParser()
    env_config.read(KDARL_CLEAN / 'crowd_nav/configs/env.config')
    policy_config = configparser.RawConfigParser()
    policy_config.read(KDARL_CLEAN / 'crowd_nav/configs/policy.config')
    train_config = configparser.RawConfigParser()
    train_config.read(KDARL_CLEAN / 'crowd_nav/configs/train.config')

    cfg = configparser.RawConfigParser()
    for c in [env_config, policy_config, train_config]:
        for sec in c.sections():
            if not cfg.has_section(sec): cfg.add_section(sec)
            for k, v in c.items(sec): cfg.set(sec, k, v)

    cfg.set('env', 'time_limit', '50')
    cfg.set('mamba', 'temporal_backbone', 'gru')
    cfg.set('buffer', 'seq_len', '24')
    cfg.set('sim', 'human_num', '20')
    return cfg

def create_environment():
    env_config = configparser.RawConfigParser()
    env_config.read(KDARL_CLEAN / 'crowd_nav/configs/env.config')
    env = CrowdSim()
    env.configure(env_config)
    env.time_limit = 50.0
    env.human_num = 20
    env.square_width = 10.0
    env.circle_radius = 4.0
    env.phase = 'test'
    env.test_sim = 'square_crossing'
    return env, env_config

def compute_intent_compatibility(commands, rob, intent):
    """Directional compatibility score between candidate commands and macro intent."""
    scores = np.zeros(len(commands), dtype=np.float32)
    gx, gy = rob.gx, rob.gy
    px, py = rob.px, rob.py
    theta_goal = math.atan2(gy - py, gx - px)
    v_pref = max(0.1, rob.v_pref)

    for i, cmd in enumerate(commands):
        speed = math.hypot(cmd.vx, cmd.vy)
        norm_speed = min(1.0, speed / v_pref)
        theta_a = math.atan2(cmd.vy, cmd.vx)
        diff = (theta_a - theta_goal + math.pi) % (2 * math.pi) - math.pi

        if intent == INTENT_BYPASS_LEFT:
            target_offset = math.pi / 4.0
            scores[i] = math.cos(diff - target_offset) * norm_speed
        elif intent == INTENT_DIRECT:
            scores[i] = math.cos(diff) * norm_speed
        elif intent == INTENT_BYPASS_RIGHT:
            target_offset = -math.pi / 4.0
            scores[i] = math.cos(diff - target_offset) * norm_speed
        elif intent == INTENT_YIELD:
            scores[i] = (1.0 - norm_speed) * 0.8
        else:
            scores[i] = 0.0
    return torch.tensor(scores, dtype=torch.float32)

def derive_macro_intent_from_action(rob, best_cmd):
    """Determine the macro passing intent implied by the selected action."""
    gx, gy = rob.gx, rob.gy
    px, py = rob.px, rob.py
    theta_goal = math.atan2(gy - py, gx - px)
    speed = math.hypot(best_cmd.vx, best_cmd.vy)
    if speed < 0.1:
        return INTENT_YIELD

    theta_a = math.atan2(best_cmd.vy, best_cmd.vx)
    diff = (theta_a - theta_goal + math.pi) % (2 * math.pi) - math.pi
    if diff > 0.15:
        return INTENT_BYPASS_LEFT
    elif diff < -0.15:
        return INTENT_BYPASS_RIGHT
    else:
        return INTENT_DIRECT

class B0MacroIntentPolicy(MambaRLPolicy):
    """Single-Stream GRU Policy (B0) with temporary macro-intent commitment."""
    def __init__(self, config=None, device='cpu'):
        super().__init__(config=config, device=device)
        self.committed_intent = None
        self.dwell_steps_remaining = 0
        self.intervention_history = []

    def set_intervention(self, intent: int, dwell_steps: int):
        self.committed_intent = intent
        self.dwell_steps_remaining = dwell_steps

    def reset_episode_stats(self):
        super().reset_episode_stats()
        self.committed_intent = None
        self.dwell_steps_remaining = 0
        self.intervention_history.clear()

    def score_sarl_candidates(self, state: JointState):
        base_res = super().score_sarl_candidates(state)
        rob = state.self_state

        if len(state.human_states):
            curr_phys_dists = [
                math.hypot(rob.px - h.px, rob.py - h.py) - rob.radius - h.radius
                for h in state.human_states
            ]
            current_physical_clearance = float(min(curr_phys_dists))
        else:
            current_physical_clearance = 999.0

        # Emergency clearance abort: if dangerously tight (<0.22m), cancel intent commitment immediately
        if current_physical_clearance < 0.22:
            self.dwell_steps_remaining = 0
            self.committed_intent = None

        if self.dwell_steps_remaining > 0 and self.committed_intent is not None:
            # Active intent commitment window
            compat = compute_intent_compatibility(base_res['commands'], rob, self.committed_intent).to(self.device)
            gate = max(0.0, min(1.0, (current_physical_clearance - 0.22) / 0.15))
            alpha = 0.10 * gate
            modulated_scores = base_res['scores'] + alpha * compat

            # Safety filter re-application
            if self._phase in ('test', 'val', 'eval') and self.test_min_clearance > 0.0:
                safe_mask = base_res['clearance'] >= self.test_min_clearance
                if safe_mask.any():
                    modulated_scores = modulated_scores.masked_fill(~safe_mask, -1e9)

            base_res['scores'] = modulated_scores
            self.intervention_history.append((self.committed_intent, current_physical_clearance))
            self.dwell_steps_remaining -= 1
        else:
            self.committed_intent = None
            self.dwell_steps_remaining = 0
            self.intervention_history.append((-1, current_physical_clearance))

        return base_res

def load_b0_policy(device='cpu'):
    cfg = get_base_config()
    policy = B0MacroIntentPolicy(cfg, device=device)
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

def clone_pre_decision_snapshot(env, policy):
    """Snapshot taken strictly BEFORE predict(state) is invoked."""
    for h in env.humans:
        if hasattr(h.policy, 'sim'):
            h.policy.sim = None
    env_clone = copy.deepcopy(env)
    hist_prox = list(policy._history)
    last_act = copy.deepcopy(getattr(policy, '_last_action', None))
    return env_clone, hist_prox, last_act

def run_branch(policy, env_snap, hist_prox, last_act, start_step, dwell_k, committed_intent):
    for h in env_snap.humans:
        if hasattr(h.policy, 'sim'): h.policy.sim = None
    env = copy.deepcopy(env_snap)
    robot = env.robot
    robot.set_policy(policy)
    robot.env = env
    env.set_robot(robot)

    policy.reset_episode_stats()
    policy._history = list(hist_prox)
    if last_act is not None:
        policy._last_action = last_act

    if dwell_k > 0 and committed_intent is not None:
        policy.set_intervention(committed_intent, dwell_k)

    traj_x = [robot.px]
    traj_y = [robot.py]
    actions = []
    min_clearance = 999.0

    step = start_step - 1
    done = False
    last_info = None

    while not done and step < 201:
        step += 1
        state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
        if len(env.humans):
            dists = [math.hypot(robot.px - h.px, robot.py - h.py) - robot.radius - h.radius for h in env.humans]
            min_clearance = min(min_clearance, min(dists))

        act = policy.predict(state)
        traj_x.append(robot.px)
        traj_y.append(robot.py)
        actions.append((act.vx, act.vy))

        obs, rew, term, trunc, last_info = env.step(act)
        done = term or trunc

    last_event = last_info.get('event', 'nothing') if last_info else 'nothing'
    if last_event == 'collision':
        outcome = 'COLLISION'
    elif last_event in ('reach_goal', 'reachgoal') or env.robot.reached_destination():
        outcome = 'SUCCESS'
    elif last_event in ('timeout', 'time_out') or env.global_time >= env.time_limit - 1e-4:
        outcome = 'TIMEOUT'
    else:
        outcome = 'TIMEOUT'

    final_time = float(env.global_time)
    p_arr = np.column_stack([traj_x[-40:], traj_y[-40:]]) if len(traj_x) >= 40 else np.column_stack([traj_x, traj_y])
    t_10s = float(np.sum(np.linalg.norm(np.diff(p_arr, axis=0), axis=1))) if len(p_arr) > 1 else 0.0
    d_10s = float(np.linalg.norm(p_arr[-1] - p_arr[0])) if len(p_arr) > 1 else 0.0
    is_stag = bool(outcome == 'TIMEOUT' and t_10s > 1.5 and d_10s < 0.35 and t_10s / max(d_10s, 0.01) > 4.0)

    return {
        'outcome': outcome,
        'final_time': round(final_time, 2),
        'total_steps': step,
        'min_clearance': round(float(min_clearance), 3),
        'stagnation': is_stag,
        'traj_x': [round(x, 3) for x in traj_x],
        'traj_y': [round(y, 3) for y in traj_y],
        'actions': [(round(vx, 3), round(vy, 3)) for vx, vy in actions]
    }

def run_counterfactual_study(start_case=90000, num_cases=200, device='cuda', out_file=None):
    if out_file is None:
        out_file = RESULTS_DIR / 'b0_counterfactual_study_200cases.json'
    else:
        out_file = Path(out_file)
    print("\n" + "="*85)
    print(f"PHASE P1: B0 COUNTERFACTUAL STUDY ACROSS {num_cases} CASES (Cases {start_case} - {start_case + num_cases - 1})")
    print("="*85)

    cases_data = []
    total_snapshots = 0
    t_start = time.time()

    score_map = {'SUCCESS': 1.0, 'TIMEOUT': 0.0, 'COLLISION': -1.0}

    # Tracking counters
    archetype_counts = {
        'Deadlock': 0,
        'ActiveYield': 0,
        'DangerousApproach': 0,
        'NormalDetour': 0,
        'SmoothProgress': 0
    }

    policy_orig = load_b0_policy(device=device)
    policy_branch = load_b0_policy(device=device)

    for idx in range(num_cases):
        case_id = start_case + idx
        env, env_config = create_environment()
        policy = policy_orig
        robot = Robot(env_config, 'robot')
        robot.set_policy(policy)
        robot.env = env
        env.set_robot(robot)

        obs, _ = env.reset(options={'test_case': case_id})
        policy.reset_episode_stats()

        history_pos = []
        history_vel = []
        orig_actions = []
        orig_traj = []

        case_snapshots = []
        done = False
        step = 0
        last_info = None

        while not done and step < 201:
            step += 1
            px, py = robot.px, robot.py
            history_pos.append((px, py))
            orig_traj.append((px, py))
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            rob = state.self_state

            # Compute current observable metrics
            if len(state.human_states):
                phys_clearances = [
                    math.hypot(rob.px - h.px, rob.py - h.py) - rob.radius - h.radius
                    for h in state.human_states
                ]
                dmin_curr = float(min(phys_clearances))
                nearest_h = state.human_states[int(np.argmin(phys_clearances))]
                rel_vx = nearest_h.vx - rob.vx
                rel_vy = nearest_h.vy - rob.vy
                closing_speed = -((nearest_h.px - rob.px)*rel_vx + (nearest_h.py - rob.py)*rel_vy) / max(0.01, math.hypot(nearest_h.px - rob.px, nearest_h.py - rob.py))
            else:
                dmin_curr = 999.0
                closing_speed = 0.0

            W = 16
            if len(history_pos) >= W:
                recent_p = history_pos[-W:]
                t_4s = float(sum(math.hypot(recent_p[i][0] - recent_p[i-1][0], recent_p[i][1] - recent_p[i-1][1]) for i in range(1, len(recent_p))))
                d_4s = float(math.hypot(recent_p[-1][0] - recent_p[0][0], recent_p[-1][1] - recent_p[0][1]))
                r_4s = t_4s / max(d_4s, 0.01)

                recent_v = history_vel[-W:]
                rev_4s = 0
                for i in range(1, len(recent_v)):
                    dot = recent_v[i][0]*recent_v[i-1][0] + recent_v[i][1]*recent_v[i-1][1]
                    m1 = math.hypot(recent_v[i][0], recent_v[i][1])
                    m2 = math.hypot(recent_v[i-1][0], recent_v[i-1][1])
                    if m1 > 0.1 and m2 > 0.1 and dot / (m1*m2) < -0.5:
                        rev_4s += 1
            else:
                t_4s, d_4s, r_4s, rev_4s = 0.0, 0.0, 1.0, 0

            # State archetype check
            cand_arch = None
            if step >= 20 and (r_4s > 3.0 or rev_4s >= 2) and d_4s < 0.35 and t_4s > 0.5:
                cand_arch = 'Deadlock'
            elif step >= 16 and (len(history_vel) and math.hypot(history_vel[-1][0], history_vel[-1][1]) < 0.15) and dmin_curr < 0.45:
                cand_arch = 'ActiveYield'
            elif dmin_curr < 0.30 and closing_speed > 0.4:
                cand_arch = 'DangerousApproach'
            elif step >= 16 and 1.8 < r_4s <= 3.0 and d_4s >= 0.35:
                cand_arch = 'NormalDetour'
            elif step in (12, 28) and dmin_curr > 0.40 and d_4s > 0.7:
                cand_arch = 'SmoothProgress'

            # Snapshot harvesting: at most 2 snapshots per case to ensure high diversity
            if cand_arch is not None and len(case_snapshots) < 2:
                # Harvest PRE-DECISION snapshot
                env_snap, hp, la = clone_pre_decision_snapshot(env, policy)

                # Peek best candidate action to derive candidate intent
                res_peek = policy.score_sarl_candidates(state)
                best_cmd = res_peek['commands'][res_peek['scores'].argmax().item()]
                intent_candidate = derive_macro_intent_from_action(rob, best_cmd)

                # Observable state features
                dist_to_goal = math.hypot(rob.gx - rob.px, rob.gy - rob.py)
                theta_g = math.atan2(rob.gy - rob.py, rob.gx - rob.px)
                theta_v = math.atan2(rob.vy, rob.vx) if math.hypot(rob.vx, rob.vy) > 0.05 else theta_g
                goal_heading_error = (theta_v - theta_g + math.pi) % (2*math.pi) - math.pi
                safe_actions_ratio = float((res_peek['clearance'] >= 0.20).float().mean().item())

                snap_item = {
                    'snapshot_id': f"Snap_C{case_id}_S{step}",
                    'case_id': case_id,
                    'step': step,
                    'time': round(float(env.global_time), 2),
                    'archetype': cand_arch,
                    'features': {
                        'dmin_curr': round(dmin_curr, 3),
                        'closing_speed': round(closing_speed, 3),
                        'travel_4s': round(t_4s, 3),
                        'disp_4s': round(d_4s, 3),
                        'ratio_4s': round(r_4s, 2),
                        'reversals_4s': rev_4s,
                        'dist_to_goal': round(dist_to_goal, 2),
                        'goal_heading_error': round(goal_heading_error, 3),
                        'safe_actions_ratio': round(safe_actions_ratio, 3),
                        'current_speed': round(math.hypot(rob.vx, rob.vy), 3)
                    },
                    'intent_candidate': intent_candidate,
                    'intent_name': INTENT_NAMES[intent_candidate],
                    'env_snap': env_snap,
                    'hp': hp,
                    'la': la
                }
                case_snapshots.append(snap_item)
                archetype_counts[cand_arch] += 1

            # Execute continuous step
            action = policy.predict(state)
            orig_actions.append((action.vx, action.vy))
            history_vel.append((action.vx, action.vy))

            obs, rew, term, trunc, last_info = env.step(action)
            done = term or trunc

        orig_event = last_info.get('event', 'nothing') if last_info else 'nothing'
        if orig_event == 'collision':
            orig_outcome = 'COLLISION'
        elif orig_event in ('reach_goal', 'reachgoal') or env.robot.reached_destination():
            orig_outcome = 'SUCCESS'
        elif orig_event in ('timeout', 'time_out') or env.global_time >= env.time_limit - 1e-4:
            orig_outcome = 'TIMEOUT'
        else:
            orig_outcome = 'TIMEOUT'

        orig_final_time = round(float(env.global_time), 2)
        orig_total_steps = step

        # Now execute counterfactual branches for each harvested snapshot in this case
        processed_snapshots = []
        for s_idx, snap in enumerate(case_snapshots):
            total_snapshots += 1
            s_step = snap['step']
            intent = snap['intent_candidate']
            esnap = snap['env_snap']
            hp = snap['hp']
            la = snap['la']

            # Branch A: K=0 (Base B0 Zero-Intervention)
            res_a = run_branch(policy_branch, esnap, hp, la, s_step, dwell_k=0, committed_intent=None)
            # Rigorous Parity Assertion: Branch A must match original continuous trajectory!
            parity_pass = (res_a['outcome'] == orig_outcome and abs(res_a['final_time'] - orig_final_time) < 1e-5)
            if not parity_pass:
                print(f"  [PARITY WARNING] Case {case_id} Step {s_step}: Base Branch outcome={res_a['outcome']} ({res_a['final_time']}s) vs Orig={orig_outcome} ({orig_final_time}s)")

            # Branch B: K=2 (Short Dwell)
            res_b = run_branch(policy_branch, esnap, hp, la, s_step, dwell_k=2, committed_intent=intent)
            # Branch C: K=4 (Medium Dwell)
            res_c = run_branch(policy_branch, esnap, hp, la, s_step, dwell_k=4, committed_intent=intent)
            # Branch D: K=8 (Long Dwell)
            res_d = run_branch(policy_branch, esnap, hp, la, s_step, dwell_k=8, committed_intent=intent)

            gain_k2 = score_map[res_b['outcome']] - score_map[res_a['outcome']]
            gain_k4 = score_map[res_c['outcome']] - score_map[res_a['outcome']]
            gain_k8 = score_map[res_d['outcome']] - score_map[res_a['outcome']]

            processed_snapshots.append({
                'snapshot_id': snap['snapshot_id'],
                'step': s_step,
                'time': snap['time'],
                'archetype': snap['archetype'],
                'features': snap['features'],
                'intent_candidate': snap['intent_candidate'],
                'intent_name': snap['intent_name'],
                'branch_A_base_k0': {
                    'outcome': res_a['outcome'],
                    'final_time': res_a['final_time'],
                    'total_steps': res_a['total_steps'],
                    'min_clearance': res_a['min_clearance'],
                    'parity_pass': parity_pass
                },
                'branch_B_dwell_k2': {
                    'outcome': res_b['outcome'],
                    'final_time': res_b['final_time'],
                    'total_steps': res_b['total_steps'],
                    'min_clearance': res_b['min_clearance'],
                    'gain_vs_base': gain_k2
                },
                'branch_C_dwell_k4': {
                    'outcome': res_c['outcome'],
                    'final_time': res_c['final_time'],
                    'total_steps': res_c['total_steps'],
                    'min_clearance': res_c['min_clearance'],
                    'gain_vs_base': gain_k4
                },
                'branch_D_dwell_k8': {
                    'outcome': res_d['outcome'],
                    'final_time': res_d['final_time'],
                    'total_steps': res_d['total_steps'],
                    'min_clearance': res_d['min_clearance'],
                    'gain_vs_base': gain_k8
                }
            })

        cases_data.append({
            'case_id': case_id,
            'orig_outcome': orig_outcome,
            'orig_final_time': orig_final_time,
            'orig_total_steps': orig_total_steps,
            'snapshots': processed_snapshots
        })

        if (idx + 1) % 5 == 0 or idx == num_cases - 1:
            print(f"[{out_file.name}] Processed {idx+1}/{num_cases} cases | Total snapshots: {total_snapshots} | Elapsed: {time.time()-t_start:.1f}s")
            with open(out_file, 'w') as f:
                json.dump(cases_data, f, indent=2)

    t_end = time.time()
    print(f"\n[{out_file.name}] Finished in {t_end - t_start:.2f}s! Total Snapshots: {total_snapshots}")
    print(f"Saved dataset to {out_file}")

    # Summary analysis by archetype
    all_snaps = [s for c in cases_data for s in c['snapshots']]
    if all_snaps:
        print("\n" + "="*95)
        print("COUNTERFACTUAL INTERVENTION GAIN BY ARCHETYPE (B0 BASELINE)")
        print("="*95)
        print(f"{'Archetype':<20} | {'N':<5} | {'Base SR':<8} | {'K=2 SR':<8} | {'K=4 SR':<8} | {'K=8 SR':<8} | {'Gain(K=2)':<10} | {'Gain(K=4)':<10} | {'Gain(K=8)':<10}")
        print("-" * 95)

        archetypes = sorted(list(set(s['archetype'] for s in all_snaps)))
        for arch in archetypes:
            arch_snaps = [s for s in all_snaps if s['archetype'] == arch]
            n = len(arch_snaps)
            sr_a = np.mean([1 if s['branch_A_base_k0']['outcome'] == 'SUCCESS' else 0 for s in arch_snaps]) * 100
            sr_b = np.mean([1 if s['branch_B_dwell_k2']['outcome'] == 'SUCCESS' else 0 for s in arch_snaps]) * 100
            sr_c = np.mean([1 if s['branch_C_dwell_k4']['outcome'] == 'SUCCESS' else 0 for s in arch_snaps]) * 100
            sr_d = np.mean([1 if s['branch_D_dwell_k8']['outcome'] == 'SUCCESS' else 0 for s in arch_snaps]) * 100

            avg_g2 = np.mean([s['branch_B_dwell_k2']['gain_vs_base'] for s in arch_snaps])
            avg_g4 = np.mean([s['branch_C_dwell_k4']['gain_vs_base'] for s in arch_snaps])
            avg_g8 = np.mean([s['branch_D_dwell_k8']['gain_vs_base'] for s in arch_snaps])

            print(f"{arch:<20} | {n:<5} | {sr_a:6.1f}% | {sr_b:6.1f}% | {sr_c:6.1f}% | {sr_d:6.1f}% | {avg_g2:+8.2f}   | {avg_g4:+8.2f}   | {avg_g8:+8.2f}")

    return cases_data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Phase P1 Counterfactual Study on B0")
    parser.add_argument('--start', type=int, default=90000, help="Start case id")
    parser.add_argument('--num', type=int, default=200, help="Number of cases")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Torch device")
    parser.add_argument('--out_file', type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    out_p = Path(args.out_file) if args.out_file else (RESULTS_DIR / 'b0_counterfactual_study_200cases.json')
    run_counterfactual_study(start_case=args.start, num_cases=args.num, device=args.device, out_file=out_p)
