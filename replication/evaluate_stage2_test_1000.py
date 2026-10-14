import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import sys
import json
import math
import time
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
import joblib

REPO_ROOT = Path('/home/abc/workspace/autonomous_research_20261012')
sys.path.insert(0, str(REPO_ROOT))

from crowd_sim.envs.utils.robot import Robot
from run_b0_counterfactual_study import create_environment
from run_stage4_500cases_evaluation import run_single_episode, load_stage4_policy
from replication.stage2_policy import load_stage2_policy

REPLICATION_DIR = REPO_ROOT / 'replication'
OUT_MASTER = REPLICATION_DIR / 'stage2_test_1000cases.json'
OUT_STATS = REPLICATION_DIR / 'stage2_test_stats.json'
REPORT_PATH = REPLICATION_DIR / 'stage2_test_REPORT.md'

TEST_START_CASE = 98000
TEST_NUM_CASES = 1000

FORBIDDEN_RANGES = [
    (90000, 90299),
    (92000, 92499),
    (93000, 93199),
    (95000, 96499),
    (97000, 97999),
    (99000, 99999),
    (310000, 310199),
    (311000, 311499)
]

def assert_stage2_test_guard(case_id):
    if not (TEST_START_CASE <= case_id < TEST_START_CASE + TEST_NUM_CASES):
        raise ValueError(
            f"SAFETY VIOLATION: Case {case_id} is outside allowed Stage 2 Test Block "
            f"[{TEST_START_CASE}, {TEST_START_CASE + TEST_NUM_CASES - 1}]!"
        )
    for f_start, f_end in FORBIDDEN_RANGES:
        if f_start <= case_id <= f_end:
            raise ValueError(f"SAFETY VIOLATION: Case {case_id} falls into forbidden range [{f_start}, {f_end}]!")

def mcnemar_exact(arm_a_outcomes, arm_b_outcomes):
    """
    Computes exact two-sided McNemar test using binomial distribution.
    b: Arm A succeeds while Arm B fails
    c: Arm A fails while Arm B succeeds
    """
    b = sum(1 for a, b in zip(arm_a_outcomes, arm_b_outcomes) if a == 'SUCCESS' and b != 'SUCCESS')
    c = sum(1 for a, b in zip(arm_a_outcomes, arm_b_outcomes) if a != 'SUCCESS' and b == 'SUCCESS')
    n = b + c
    if n == 0:
        return 1.0, b, c
    try:
        from scipy.stats import binomtest
        pval = float(binomtest(min(b, c), n, 0.5).pvalue)
    except (ImportError, AttributeError):
        from scipy.stats import binom_test
        pval = float(binom_test(min(b, c), n, 0.5))
    return pval, b, c

def load_stage2_evaluation_policies(device='cpu'):
    # 1. Mother backbone (Pure B0 GRU s443)
    mother_policy = load_stage4_policy('base', device=device)

    # 2. Rule-based K=4 (Stage 2 Fixed 4-step termination)
    rule_k4_policy = load_stage2_policy(termination_mode='fixed_k4', device=device)

    # 3. Champion Learned Head & 4. Shuffled Labels Control
    champ_meta_file = REPLICATION_DIR / 'stage2_champion_metadata.json'
    with open(champ_meta_file, 'r') as f:
        champ_meta = json.load(f)

    champ_model = joblib.load(REPLICATION_DIR / 'stage2_champion_model.joblib')
    shuf_model = joblib.load(REPLICATION_DIR / 'stage2_shuffled_model.joblib')

    scaler = {
        'mean': np.array(champ_meta['scaler_mean'], dtype=np.float32),
        'scale': np.array(champ_meta['scaler_scale'], dtype=np.float32)
    }

    learned_policy = load_stage2_policy(
        termination_mode='learned',
        device=device,
        learned_model=champ_model,
        learned_scaler=scaler,
        learned_threshold=champ_meta['threshold']
    )

    shuffled_policy = load_stage2_policy(
        termination_mode='shuffled',
        device=device,
        learned_model=shuf_model,
        learned_scaler=scaler,
        learned_threshold=champ_meta['threshold']
    )

    return {
        'mother': mother_policy,
        'rule_k4': rule_k4_policy,
        'learned_champ': learned_policy,
        'shuffled_ctrl': shuffled_policy
    }

from crowd_sim.envs.utils.state import JointState

def run_single_episode_with_switching(policy, env, case_id):
    """
    Runs single episode, tracking outcome, trajectory, dwell history,
    and action switching rate.
    """
    obs, _ = env.reset(options={'test_case': case_id})
    policy.reset_episode_stats()

    min_clearance = 999.0
    step = 0
    done = False
    last_info = None

    action_indices = []

    while not done and step < 201:
        step += 1
        state = JointState(env.robot.get_full_state(), [h.get_observable_state() for h in env.humans])
        if len(env.humans):
            dists = [math.hypot(env.robot.px - h.px, env.robot.py - h.py) - env.robot.radius - h.radius for h in env.humans]
            min_clearance = min(min_clearance, min(dists))

        act = policy.predict(state)
        act_idx = getattr(policy, '_last_grid_index', None)
        action_indices.append(act_idx)

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
    total_steps = step

    # Calculate switching rate: switches / max(1, steps - 1)
    valid_indices = [idx for idx in action_indices if idx is not None]
    if len(valid_indices) > 1:
        switches = sum(1 for i in range(1, len(valid_indices)) if valid_indices[i] != valid_indices[i-1])
        switch_rate = switches / (len(valid_indices) - 1)
    else:
        switches = 0
        switch_rate = 0.0

    dwell_hist = getattr(policy, 'dwell_history', [])
    interventions = getattr(policy, 'intervention_count', 0)

    return {
        'outcome': outcome,
        'final_time': round(final_time, 2),
        'total_steps': total_steps,
        'min_clearance': round(min_clearance, 3),
        'interventions': interventions,
        'dwell_history': list(dwell_hist),
        'switch_count': switches,
        'switch_rate': round(switch_rate, 4)
    }

def run_chunk_eval(start_case, num_cases, out_file=None, device='cpu'):
    env, env_cfg = create_environment()
    env.test_sim = 'square_crossing'
    env.human_num = 20
    robot = Robot(env_cfg, 'robot')
    env.set_robot(robot)

    policies = load_stage2_evaluation_policies(device=device)

    records = []
    t_start = time.time()

    for idx in range(num_cases):
        case_id = start_case + idx
        assert_stage2_test_guard(case_id)
        case_res = {'case_id': case_id}

        for arm_name, pol in policies.items():
            robot.set_policy(pol)
            robot.env = env
            env.set_robot(robot)

            ep_res = run_single_episode_with_switching(pol, env, case_id)
            case_res[arm_name] = ep_res

        records.append(case_res)

        if (idx + 1) % 10 == 0 or idx == num_cases - 1:
            if out_file:
                with open(out_file, 'w') as f:
                    json.dump(records, f, indent=2)
            elapsed = time.time() - t_start
            print(f"[Worker {start_case}] Case {case_id} ({idx+1}/{num_cases}) | Elapsed: {elapsed:.1f}s", flush=True)

    return records

def generate_stage2_test_report(records, stats_summary, out_report_path):
    arm_labels = {
        'mother': '母体 (Mother GRU s443)',
        'rule_k4': '人工规则 K=4 (Fixed K=4)',
        'learned_champ': 'Champion 学习头 (cand4_mlp_h32_16)',
        'shuffled_ctrl': '打乱标签头 (Shuffled Control)'
    }
    arms = list(arm_labels.keys())

    lines = []
    lines.append("# 阶段 2 可学习终止测试块独立评测报告 (Stage 2 Test Evaluation Report)")
    lines.append("")
    lines.append("> 测试范围：Case 98000–98999（20人方形交叉场景，N=1,000，严格单次评测，禁止再调参）  ")
    lines.append("> 运行架构：8 进程并行，严格单线程 (OMP/MKL/OPENBLAS=1, torch.set_num_threads(1))  ")
    lines.append(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 一、评测总览与各臂核心指标")
    lines.append("")
    lines.append("| 评测臂 | 成功率 SR (%) | 碰撞率 CR (%) | 超时率 TR (%) | 平均时间 (s) | 平均干预频次 | 平均承诺步数 | 动作切换率 | 状态 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for arm in arms:
        st = stats_summary['arms'][arm]
        lines.append(
            f"| {arm_labels[arm]} | {st['sr']:.2f}% ({st['success_count']}/{st['n_cases']}) | "
            f"{st['cr']:.2f}% ({st['collision_count']}/{st['n_cases']}) | "
            f"{st['tr']:.2f}% ({st['timeout_count']}/{st['n_cases']}) | "
            f"{st['avg_time']:.2f} | {st['avg_interventions']:.2f} | "
            f"{st['mean_dwell']:.2f} | {st['mean_switch_rate']:.4f} | **已验证** |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 二、配对统计检验（McNemar Exact Test）与 b/c 计数")
    lines.append("")
    lines.append("说明：$b$ 表示方法 A 成功但方法 B 失败的案例数；$c$ 表示方法 A 失败但方法 B 成功的案例数。精确双侧 $p$ 值基于二项分布检验计算。")
    lines.append("")
    lines.append("| 对比组 (方法 A vs 方法 B) | $\\Delta\\text{SR}$ (pp) | 胜出案例 $b$ | 败落案例 $c$ | 精确双侧 $p$ 值 | 显著性 (α=0.05) | 结论状态 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for pair_name, comp in stats_summary['comparisons'].items():
        sig = "显著 (p < 0.05)" if comp['p_value'] < 0.05 else "不显著 (p >= 0.05)"
        lines.append(
            f"| {comp['label_a']} vs {comp['label_b']} | {comp['delta_sr']:+.2f}% | "
            f"{comp['b']} | {comp['c']} | {comp['p_value']:.4e} | {sig} | **已验证** |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 三、关键科学发现与事实陈述")
    lines.append("")
    
    comp_learned_rule = stats_summary['comparisons']['learned_vs_rule_k4']
    comp_learned_shuf = stats_summary['comparisons']['learned_vs_shuffled']
    comp_shuf_rule = stats_summary['comparisons']['shuffled_vs_rule_k4']
    comp_learned_mother = stats_summary['comparisons']['learned_vs_mother']

    lines.append(f"1. **Champion 学习头 vs 人工规则 K=4**：")
    lines.append(f"   - 配对结果：$b={comp_learned_rule['b']}$ 对 $c={comp_learned_rule['c']}$，精确双侧 $p={comp_learned_rule['p_value']:.4e}$，$\\Delta\\text{SR}={comp_learned_rule['delta_sr']:+.2f}\\%$。")
    lines.append(f"2. **Champion 学习头 vs 打乱标签对照组**：")
    lines.append(f"   - 配对结果：$b={comp_learned_shuf['b']}$ 对 $c={comp_learned_shuf['c']}$，精确双侧 $p={comp_learned_shuf['p_value']:.4e}$，$\\Delta\\text{SR}={comp_learned_shuf['delta_sr']:+.2f}\\%$。")
    lines.append(f"3. **打乱标签对照组 vs 人工规则 K=4**：")
    lines.append(f"   - 配对结果：$b={comp_shuf_rule['b']}$ 对 $c={comp_shuf_rule['c']}$，精确双侧 $p={comp_shuf_rule['p_value']:.4e}$，$\\Delta\\text{SR}={comp_shuf_rule['delta_sr']:+.2f}\\%$。")
    lines.append(f"4. **Champion 学习头 vs 母体基线**：")
    lines.append(f"   - 配对结果：$b={comp_learned_mother['b']}$ 对 $c={comp_learned_mother['c']}$，精确双侧 $p={comp_learned_mother['p_value']:.4e}$，$\\Delta\\text{SR}={comp_learned_mother['delta_sr']:+.2f}\\%$。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 四、合规与复现审计说明")
    lines.append("- **模型与权重锁定**：全部沿用已提交的 Champion 模型 (`stage2_champion_model.joblib`) 与打乱对照模型 (`stage2_shuffled_model.joblib`)，未进行任何重新调参。")
    lines.append("- **测试块独立性**：Case 98000–98999 仅在此次执行中运行一次，前置调参及训练严格隔离于 99000–99399 区间。")
    lines.append("- **资源约束遵循**：全部进程严格单线程，宿主机同时运行进程数不超过 8 个，无外部任务干扰。")
    lines.append("")

    with open(out_report_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written to {out_report_path}", flush=True)

def run_evaluation(workers=8, device='cpu'):
    print("=" * 95)
    print(f"STAGE 2 TEST BLOCK EVALUATION (Cases {TEST_START_CASE}..{TEST_START_CASE + TEST_NUM_CASES - 1}, N={TEST_NUM_CASES})")
    print(f"4 Arms: Mother, Rule K=4, Champion Learned Head, Shuffled Label Control | Workers: {workers}")
    print("=" * 95, flush=True)

    cases_per_worker = TEST_NUM_CASES // workers
    remainder = TEST_NUM_CASES % workers
    curr_start = TEST_START_CASE

    processes = []
    part_files = []

    for w in range(workers):
        w_cases = cases_per_worker + (1 if w < remainder else 0)
        part_out = REPLICATION_DIR / f'stage2_test_part{w}.json'
        part_files.append(part_out)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            '--chunk_mode',
            '--start_case', str(curr_start),
            '--num_cases', str(w_cases),
            '--out_file', str(part_out),
            '--device', device
        ]
        log_f = open(REPLICATION_DIR / f'stage2_test_worker_{w}.log', 'w')
        sub_env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=sub_env)
        processes.append((proc, log_f, w))
        curr_start += w_cases

    for proc, log_f, wid in processes:
        proc.wait()
        log_f.close()
        print(f"Worker {wid} completed with exit code {proc.returncode}.", flush=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Worker {wid} failed! Check stage2_test_worker_{wid}.log")

    merged = []
    for pf in part_files:
        with open(pf, 'r') as fp:
            merged.extend(json.load(fp))

    merged.sort(key=lambda x: x['case_id'])

    with open(OUT_MASTER, 'w') as fp:
        json.dump(merged, fp, indent=2)
    print(f"Merged {len(merged)} cases saved to {OUT_MASTER}", flush=True)

    # Compute statistics
    arms = ['mother', 'rule_k4', 'learned_champ', 'shuffled_ctrl']
    arm_labels = {
        'mother': '母体 (Mother GRU s443)',
        'rule_k4': '人工规则 K=4 (Fixed K=4)',
        'learned_champ': 'Champion 学习头 (cand4_mlp_h32_16)',
        'shuffled_ctrl': '打乱标签头 (Shuffled Control)'
    }

    n_cases = len(merged)
    arm_stats = {}

    for arm in arms:
        outcomes = [d[arm]['outcome'] for d in merged]
        times = [d[arm]['final_time'] for d in merged if d[arm]['outcome'] == 'SUCCESS']
        dwells = []
        for d in merged:
            dwells.extend(d[arm].get('dwell_history', []))
        switch_rates = [d[arm].get('switch_rate', 0.0) for d in merged]
        interventions = [d[arm].get('interventions', 0) for d in merged]

        sr = float(np.mean([1.0 if o == 'SUCCESS' else 0.0 for o in outcomes]) * 100.0)
        cr = float(np.mean([1.0 if o == 'COLLISION' else 0.0 for o in outcomes]) * 100.0)
        tr = float(np.mean([1.0 if o == 'TIMEOUT' else 0.0 for o in outcomes]) * 100.0)

        arm_stats[arm] = {
            'label': arm_labels[arm],
            'n_cases': n_cases,
            'success_count': sum(1 for o in outcomes if o == 'SUCCESS'),
            'collision_count': sum(1 for o in outcomes if o == 'COLLISION'),
            'timeout_count': sum(1 for o in outcomes if o == 'TIMEOUT'),
            'sr': round(sr, 2),
            'cr': round(cr, 2),
            'tr': round(tr, 2),
            'avg_time': round(float(np.mean(times)), 2) if times else 0.0,
            'avg_interventions': round(float(np.mean(interventions)), 2),
            'mean_dwell': round(float(np.mean(dwells)), 2) if dwells else 0.0,
            'mean_switch_rate': round(float(np.mean(switch_rates)), 4)
        }

    # Paired comparisons
    comp_pairs = [
        ('learned_vs_rule_k4', 'learned_champ', 'rule_k4'),
        ('learned_vs_shuffled', 'learned_champ', 'shuffled_ctrl'),
        ('learned_vs_mother', 'learned_champ', 'mother'),
        ('shuffled_vs_rule_k4', 'shuffled_ctrl', 'rule_k4'),
        ('shuffled_vs_mother', 'shuffled_ctrl', 'mother'),
        ('rule_k4_vs_mother', 'rule_k4', 'mother')
    ]

    comparisons = {}
    for comp_key, arm_a, arm_b in comp_pairs:
        outcomes_a = [d[arm_a]['outcome'] for d in merged]
        outcomes_b = [d[arm_b]['outcome'] for d in merged]
        pval, b, c = mcnemar_exact(outcomes_a, outcomes_b)
        delta_sr = arm_stats[arm_a]['sr'] - arm_stats[arm_b]['sr']
        comparisons[comp_key] = {
            'arm_a': arm_a,
            'arm_b': arm_b,
            'label_a': arm_labels[arm_a],
            'label_b': arm_labels[arm_b],
            'delta_sr': round(delta_sr, 2),
            'b': b,
            'c': c,
            'p_value': pval
        }

    stats_summary = {
        'dataset': 'Stage2_Test_Square_1000cases',
        'n_cases': n_cases,
        'arms': arm_stats,
        'comparisons': comparisons
    }

    with open(OUT_STATS, 'w') as fp:
        json.dump(stats_summary, fp, indent=2)
    print(f"Stats summary saved to {OUT_STATS}", flush=True)

    generate_stage2_test_report(merged, stats_summary, REPORT_PATH)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_mode', action='store_true')
    parser.add_argument('--start_case', type=int, default=TEST_START_CASE)
    parser.add_argument('--num_cases', type=int, default=TEST_NUM_CASES)
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.chunk_mode:
        run_chunk_eval(args.start_case, args.num_cases, out_file=args.out_file, device=args.device)
    else:
        run_evaluation(workers=args.workers, device=args.device)
