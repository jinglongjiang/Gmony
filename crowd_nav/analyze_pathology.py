#!/usr/bin/env python3
"""
分析是否存在"宁愿超时也不冒险碰撞"的畸形策略
"""
import re
import numpy as np
from typing import List, Tuple

def parse_trajectory_log(log_path: str):
    """解析trajectory.log，提取成功和超时episode的轨迹数据"""
    with open(log_path, 'r') as f:
        content = f.read()

    # 分离每个episode
    episodes = content.strip().split('\n\n')

    success_episodes = []
    timeout_episodes = []

    for ep_text in episodes:
        if not ep_text.strip():
            continue

        lines = ep_text.strip().split('\n')
        if len(lines) < 2:
            continue

        # 解析header
        header = lines[0]
        match = re.search(r'START=\(([-\d.]+),\s*([-\d.]+)\)\s+GOAL=\(([-\d.]+),\s*([-\d.]+)\)\s+RESULT=(\w+)\s+STEPS=(\d+)', header)
        if not match:
            continue

        sx, sy, gx, gy, result, steps = match.groups()
        sx, sy, gx, gy = float(sx), float(sy), float(gx), float(gy)
        steps = int(steps)

        # 解析trajectory
        traj_line = lines[1] if len(lines) > 1 else ""
        if not traj_line.startswith("TRAJECTORY:"):
            continue

        traj_text = traj_line.replace("TRAJECTORY:", "").strip()
        positions = []
        for pos_match in re.finditer(r'\(([-\d.]+),\s*([-\d.]+)\)', traj_text):
            x, y = pos_match.groups()
            positions.append((float(x), float(y)))

        if len(positions) < 2:
            continue

        episode_data = {
            'start': (sx, sy),
            'goal': (gx, gy),
            'result': result,
            'steps': steps,
            'positions': positions
        }

        if result == 'SUCCESS':
            success_episodes.append(episode_data)
        elif result == 'TIMEOUT':
            timeout_episodes.append(episode_data)

    return success_episodes, timeout_episodes

def calculate_speed_profile(episode):
    """计算episode的速度分布"""
    positions = episode['positions']
    speeds = []

    for i in range(len(positions) - 1):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        speed = dist / 0.25  # dt=0.25s
        speeds.append(speed)

    return speeds

def calculate_progress(episode):
    """计算距离目标的进度"""
    positions = episode['positions']
    gx, gy = episode['goal']

    distances_to_goal = []
    for x, y in positions:
        dist = np.sqrt((x - gx)**2 + (y - gy)**2)
        distances_to_goal.append(dist)

    initial_dist = distances_to_goal[0]
    final_dist = distances_to_goal[-1]
    min_dist = min(distances_to_goal)

    return initial_dist, final_dist, min_dist, distances_to_goal

def detect_hesitation(speeds):
    """检测是否有"犹豫"行为（频繁减速到接近0）"""
    low_speed_threshold = 0.1  # m/s
    low_speed_count = sum(1 for s in speeds if s < low_speed_threshold)
    low_speed_ratio = low_speed_count / len(speeds) if speeds else 0

    # 检测"停顿"次数（连续低速）
    stop_events = 0
    in_stop = False
    for s in speeds:
        if s < low_speed_threshold:
            if not in_stop:
                stop_events += 1
                in_stop = True
        else:
            in_stop = False

    return low_speed_ratio, stop_events

def main():
    log_path = "runs/mamba_vl/trajectory.log"

    print("=" * 80)
    print("畸形策略分析报告")
    print("=" * 80)
    print()

    success_eps, timeout_eps = parse_trajectory_log(log_path)

    print(f"总样本: {len(success_eps)} SUCCESS, {len(timeout_eps)} TIMEOUT")
    print()

    # 分析速度特征
    print("📊 速度特征对比")
    print("-" * 80)

    success_speeds = []
    timeout_speeds = []

    for ep in success_eps:
        speeds = calculate_speed_profile(ep)
        success_speeds.extend(speeds)

    for ep in timeout_eps:
        speeds = calculate_speed_profile(ep)
        timeout_speeds.extend(speeds)

    if success_speeds:
        print(f"SUCCESS平均速度: {np.mean(success_speeds):.3f} m/s (±{np.std(success_speeds):.3f})")
        print(f"SUCCESS中位速度: {np.median(success_speeds):.3f} m/s")
        print(f"SUCCESS低速比例: {sum(1 for s in success_speeds if s < 0.1) / len(success_speeds):.1%} (v < 0.1 m/s)")

    if timeout_speeds:
        print(f"TIMEOUT平均速度: {np.mean(timeout_speeds):.3f} m/s (±{np.std(timeout_speeds):.3f})")
        print(f"TIMEOUT中位速度: {np.median(timeout_speeds):.3f} m/s")
        print(f"TIMEOUT低速比例: {sum(1 for s in timeout_speeds if s < 0.1) / len(timeout_speeds):.1%} (v < 0.1 m/s)")

    print()

    # 分析犹豫行为
    print("🛑 犹豫行为检测")
    print("-" * 80)

    success_hesitations = []
    timeout_hesitations = []

    for ep in success_eps[:50]:  # 取前50个样本
        speeds = calculate_speed_profile(ep)
        low_ratio, stop_events = detect_hesitation(speeds)
        success_hesitations.append((low_ratio, stop_events))

    for ep in timeout_eps[:50]:
        speeds = calculate_speed_profile(ep)
        low_ratio, stop_events = detect_hesitation(speeds)
        timeout_hesitations.append((low_ratio, stop_events))

    if success_hesitations:
        success_low_ratios = [h[0] for h in success_hesitations]
        success_stops = [h[1] for h in success_hesitations]
        print(f"SUCCESS平均低速占比: {np.mean(success_low_ratios):.1%}")
        print(f"SUCCESS平均停顿次数: {np.mean(success_stops):.1f}")

    if timeout_hesitations:
        timeout_low_ratios = [h[0] for h in timeout_hesitations]
        timeout_stops = [h[1] for h in timeout_hesitations]
        print(f"TIMEOUT平均低速占比: {np.mean(timeout_low_ratios):.1%}")
        print(f"TIMEOUT平均停顿次数: {np.mean(timeout_stops):.1f}")

    print()

    # 分析目标接近度
    print("🎯 目标进度分析")
    print("-" * 80)

    timeout_progress = []
    for ep in timeout_eps[:50]:
        init_dist, final_dist, min_dist, _ = calculate_progress(ep)
        progress = (init_dist - final_dist) / init_dist if init_dist > 0 else 0
        timeout_progress.append({
            'init': init_dist,
            'final': final_dist,
            'min': min_dist,
            'progress': progress
        })

    if timeout_progress:
        avg_init = np.mean([p['init'] for p in timeout_progress])
        avg_final = np.mean([p['final'] for p in timeout_progress])
        avg_min = np.mean([p['min'] for p in timeout_progress])
        avg_progress = np.mean([p['progress'] for p in timeout_progress])

        print(f"TIMEOUT初始距离: {avg_init:.2f}m")
        print(f"TIMEOUT结束距离: {avg_final:.2f}m")
        print(f"TIMEOUT最小距离: {avg_min:.2f}m")
        print(f"TIMEOUT平均进度: {avg_progress:.1%}")

        # 判断是否在接近目标但未成功
        close_but_timeout = sum(1 for p in timeout_progress if p['min'] < 1.0)
        print(f"曾接近目标(<1.0m)但超时: {close_but_timeout}/{len(timeout_progress)} ({close_but_timeout/len(timeout_progress):.1%})")

    print()

    # ============= 关键判断 =============
    print("=" * 80)
    print("🔍 病理诊断")
    print("=" * 80)

    # 判断标准
    pathology_indicators = []

    # 1. 超时集的速度是否显著低于成功集？
    if success_speeds and timeout_speeds:
        speed_diff = np.mean(timeout_speeds) - np.mean(success_speeds)
        if speed_diff < -0.15:  # 超时集慢0.15 m/s以上
            pathology_indicators.append(("速度过低", f"TIMEOUT比SUCCESS慢 {-speed_diff:.2f} m/s"))

    # 2. 超时集是否有更多低速比例？
    if success_speeds and timeout_speeds:
        success_low_pct = sum(1 for s in success_speeds if s < 0.1) / len(success_speeds)
        timeout_low_pct = sum(1 for s in timeout_speeds if s < 0.1) / len(timeout_speeds)
        if timeout_low_pct > success_low_pct * 1.5:
            pathology_indicators.append(("频繁低速", f"TIMEOUT低速占比 {timeout_low_pct:.1%} vs SUCCESS {success_low_pct:.1%}"))

    # 3. 超时集是否有更多停顿？
    if success_hesitations and timeout_hesitations:
        success_avg_stops = np.mean([h[1] for h in success_hesitations])
        timeout_avg_stops = np.mean([h[1] for h in timeout_hesitations])
        if timeout_avg_stops > success_avg_stops * 1.5:
            pathology_indicators.append(("频繁停顿", f"TIMEOUT平均停顿 {timeout_avg_stops:.1f}次 vs SUCCESS {success_avg_stops:.1f}次"))

    # 4. 超时集是否接近目标但未完成？
    if timeout_progress:
        close_ratio = sum(1 for p in timeout_progress if p['min'] < 1.0) / len(timeout_progress)
        if close_ratio > 0.3:
            pathology_indicators.append(("接近但放弃", f"{close_ratio:.1%}超时案例曾接近目标<1m"))

    # 输出诊断结果
    if pathology_indicators:
        print("⚠️  检测到畸形策略特征:")
        print()
        for i, (symptom, evidence) in enumerate(pathology_indicators, 1):
            print(f"  {i}. {symptom}: {evidence}")

        print()
        print("🔴 结论: 机器人表现出**过度保守**行为，宁愿超时也不冒险接近目标")
        print()
        print("🔧 建议修复方案:")
        print("  1. 调整penalty比例: collision=-0.25 vs timeout=-0.5 (当前2:1比例可能过重)")
        print("     → 建议: collision=-0.5, timeout=-0.25 (反转，让超时更不划算)")
        print("  2. 增加进度奖励: 添加distance_to_goal_reduction奖励项")
        print("  3. 添加速度惩罚: 对持续低速(<0.2 m/s)进行小负奖励")
        print("  4. 减少IL影响: 降低lambda_il或完全移除IL distillation loss")
    else:
        print("✅ 未检测到明显畸形策略特征")
        print()
        print("超时案例可能是合理的困难场景（例如人群完全堵住路径）")
        print("0% collision率说明避障策略本身是有效的")

if __name__ == "__main__":
    main()
