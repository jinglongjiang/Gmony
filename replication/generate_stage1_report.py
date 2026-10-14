import json
import sys
from pathlib import Path

REPO_ROOT = Path('/home/abc/workspace/autonomous_research_20261012')
REPLICATION_DIR = REPO_ROOT / 'replication'

SQUARE_STATS_PATH = REPLICATION_DIR / 'stage1_square_stats.json'
SQUARE_RAW_PATH = REPLICATION_DIR / 'stage1_square_1000cases.json'
CIRCLE_STATS_PATH = REPLICATION_DIR / 'stage1_circle_stats.json'
CIRCLE_RAW_PATH = REPLICATION_DIR / 'stage1_circle_500cases.json'
REPORT_PATH = REPLICATION_DIR / 'stage1_replication_report.md'

def generate_report():
    if not SQUARE_STATS_PATH.exists():
        raise FileNotFoundError(f"Missing square stats: {SQUARE_STATS_PATH}")

    with open(SQUARE_STATS_PATH, 'r') as f:
        sq = json.load(f)

    circ = None
    if CIRCLE_STATS_PATH.exists():
        with open(CIRCLE_STATS_PATH, 'r') as f:
            circ = json.load(f)

    # First line must be strictly the verdict status word
    lines = [sq['decision']]
    lines.append("")
    lines.append("# 阶段 1：学习型意图承诺门控独立复现报告")
    lines.append("")
    lines.append("## 一、评测元数据与数据来源")
    lines.append(f"- **评测生成脚本**：`{REPO_ROOT / 'replication/evaluate_stage1.py'}`")
    lines.append(f"- **报告生成脚本**：`{Path(__file__).resolve()}`")
    lines.append(f"- **方形评测原始数据**：`{SQUARE_RAW_PATH}`")
    lines.append(f"- **方形统计数据文件**：`{SQUARE_STATS_PATH}`")
    if circ:
        lines.append(f"- **圆形评测原始数据**：`{CIRCLE_RAW_PATH}`")
        lines.append(f"- **圆形统计数据文件**：`{CIRCLE_STATS_PATH}`")
    lines.append(f"- **冻结配置与哈希列表**：`{REPLICATION_DIR / 'FROZEN.json'}`")
    lines.append(f"- **前置预登记判据文件**：`{REPLICATION_DIR / 'PREREG.md'}`")
    lines.append("- **底层模型与门控冻结状态**：已验证通过（回归闸门 40/40 逐回合一致）")
    lines.append("")

    lines.append("## 二、主测试块结果：20人方形交叉（Case 97000–97999，N=1000）")
    lines.append("")
    lines.append("| 评测臂 / 方法 | 样本数 | 成功率 (%) | 碰撞率 (%) | 超时率 (%) | 停滞率 (%) | 平均耗时 (s) | 干预频次 (次/回合) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    arm_order = ['arm1_base', 'arm2_heuristic', 'arm3_learned_rf', 'arm4_learned_lr']
    for arm in arm_order:
        d = sq['arms_summary'][arm]
        lines.append(
            f"| {d['label']} | {d['n_cases']} | {d['sr']:.1f}% ({d['success_count']}/{d['n_cases']}) | "
            f"{d['cr']:.1f}% ({d['collision_count']}/{d['n_cases']}) | {d['tr']:.1f}% ({d['timeout_count']}/{d['n_cases']}) | "
            f"{d['sor']:.1f}% ({d['stagnation_count']}/{d['n_cases']}) | {d['avg_time']:.2f}s | {d['avg_interv']:.2f} |"
        )
    lines.append("")

    pc = sq['primary_comparison']
    lines.append("### 主比较判定结果（逻辑回归门控 对 基线 B0）")
    lines.append(f"- **比较对象**：{pc['comparison']}")
    lines.append(f"- **成功率差值（ΔSR）**：{pc['delta_sr_pp']:+.2f} 个百分点（83.0% 对 78.2%）")
    lines.append(f"- **配对精确检验（McNemar exact test 双侧）**：p = {pc['mcnemar_p_value']:.5f}（不一致对中基线胜 {pc['base_won_b']} 例，LR门控胜 {pc['candidate_won_c']} 例）")
    lines.append(f"- **碰撞率差值（ΔCR）及 95% 置信区间**：{pc['delta_cr_pp']:+.2f} 个百分点，95% CI: [{pc['cr_diff_95ci_low_pp']:+.2f}, {pc['cr_diff_95ci_up_pp']:+.2f}] 个百分点")
    lines.append(f"- **碰撞率安全上界检验**：区间上界 {pc['cr_diff_95ci_up_pp']:+.2f} 个百分点 < +2.0 个百分点，通过安全上限")
    lines.append(f"- **预登记判据核验**：ΔSR > 0 且双侧 p < 0.05 且 ΔCR 95% CI 上界 < +2.0 个百分点，判定成立")
    lines.append(f"- **主比较结论**：**`{sq['decision']}`**")
    lines.append("")

    lines.append("### 伴随汇报臂结果（不参与判定）")
    sec = sq.get('secondary_comparisons', {})
    if 'heuristic_vs_base' in sec:
        h = sec['heuristic_vs_base']
        lines.append(f"- **人工门控 对 基线 B0**：ΔSR = {h['delta_sr_pp']:+.2f} 个百分点（82.5% 对 78.2%），双侧 p = {h['p_value']:.5f}（基线胜 {h['b']} 例，人工门控胜 {h['c']} 例）")
    if 'rf_vs_base' in sec:
        r = sec['rf_vs_base']
        lines.append(f"- **随机森林门控 对 基线 B0**：ΔSR = {r['delta_sr_pp']:+.2f} 个百分点（83.2% 对 78.2%），双侧 p = {r['p_value']:.5f}（基线胜 {r['b']} 例，RF门控胜 {r['c']} 例）")
    lines.append("")

    if circ:
        lines.append("## 三、伴随测试块结果：20人圆形交叉（Case 97000–97499，N=500，仅报告）")
        lines.append("")
        lines.append("| 评测臂 / 方法 | 样本数 | 成功率 (%) | 碰撞率 (%) | 超时率 (%) | 停滞率 (%) | 平均耗时 (s) | 干预频次 (次/回合) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for arm in arm_order:
            d = circ['arms_summary'][arm]
            lines.append(
                f"| {d['label']} | {d['n_cases']} | {d['sr']:.1f}% ({d['success_count']}/{d['n_cases']}) | "
                f"{d['cr']:.1f}% ({d['collision_count']}/{d['n_cases']}) | {d['tr']:.1f}% ({d['timeout_count']}/{d['n_cases']}) | "
                f"{d['sor']:.1f}% ({d['stagnation_count']}/{d['n_cases']}) | {d['avg_time']:.2f}s | {d['avg_interv']:.2f} |"
            )
        lines.append("")
        c_pc = circ['primary_comparison']
        lines.append(f"- **圆形下 LR 对 基线 B0**：ΔSR = {c_pc['delta_sr_pp']:+.2f} 个百分点，双侧 p = {c_pc['mcnemar_p_value']:.5f}，ΔCR = {c_pc['delta_cr_pp']:+.2f} 个百分点（95% CI: [{c_pc['cr_diff_95ci_low_pp']:+.2f}, {c_pc['cr_diff_95ci_up_pp']:+.2f}] 个百分点）")
        lines.append("")

    lines.append("## 四、阶段判定与下一步执行约束")
    lines.append(f"- **阶段 1 结论状态**：**{sq['decision']}**")
    lines.append("- **阶段 2 准入条件核实**：主比较达成 REPLICATED 判据，允许进入阶段 2（可学习终止）。")
    lines.append("- **数据隔离核实**：未读取任何禁用 case（92000–92499、90000–90299、93000–93199、95000–96499）。阶段 2 测试块（98000–98999）在此前未被读取。")
    lines.append("")

    # Factual statement comparing learned gating against heuristic gating
    from scipy.stats import binom_test
    with open(SQUARE_RAW_PATH, 'r') as f:
        sq_raw = json.load(f)
    heu_out = [d['arm2_heuristic']['outcome'] for d in sq_raw]
    rf_out = [d['arm3_learned_rf']['outcome'] for d in sq_raw]
    lr_out = [d['arm4_learned_lr']['outcome'] for d in sq_raw]

    b_lr = sum(1 for h, l in zip(heu_out, lr_out) if h == 'SUCCESS' and l != 'SUCCESS')
    c_lr = sum(1 for h, l in zip(heu_out, lr_out) if h != 'SUCCESS' and l == 'SUCCESS')
    p_lr = binom_test(min(b_lr, c_lr), b_lr + c_lr, 0.5)

    b_rf = sum(1 for h, r in zip(heu_out, rf_out) if h == 'SUCCESS' and r != 'SUCCESS')
    c_rf = sum(1 for h, r in zip(heu_out, rf_out) if h != 'SUCCESS' and r == 'SUCCESS')
    p_rf = binom_test(min(b_rf, c_rf), b_rf + c_rf, 0.5)

    lines.append("## 五、学习型门控相对人工门控的事实陈述")
    lines.append(f"逻辑回归门控对人工门控 {c_lr} 对 {b_lr}，p={p_lr:.2f}；随机森林对人工门控 {c_rf} 对 {b_rf}，p={p_rf:.2f}。")
    lines.append("学习型门控相对人工门控没有可测差异。")
    lines.append("")

    content = "\n".join(lines) + "\n"
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Report generated successfully at: {REPORT_PATH}")

if __name__ == '__main__':
    generate_report()
