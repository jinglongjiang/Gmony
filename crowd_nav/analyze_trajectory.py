#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Analysis Tool - 分析trajectory.log中的导航质量

功能：
1. 路径长度 vs 最优路径（直线距离）
2. 平滑度（角度变化/转向频率）
3. 路径偏差（距离直线的平均偏离）
4. 时间效率
5. 成功/超时统计

用法：
    python analyze_trajectory.py --log runs/mamba_vl/trajectory.log
    python analyze_trajectory.py --log runs/mamba_vl/trajectory.log --top 10  # 分析前10条
    python analyze_trajectory.py --log runs/mamba_vl/trajectory.log --plot     # 生成可视化
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import defaultdict
from scipy.interpolate import make_interp_spline


def parse_trajectory_log(log_path):
    """
    解析trajectory.log文件

    支持两种格式：
    1. 旧格式（只有机器人）：
       [EP=1] START=(x,y) GOAL=(x,y) RESULT=SUCCESS STEPS=n
       TRAJECTORY: (x1, y1) (x2, y2) ...

    2. 新格式（机器人+行人）：
       [EP=1] START=(x,y) GOAL=(x,y) RESULT=SUCCESS STEPS=n HUMANS=3
       ROBOT_TRAJ: (x1, y1) (x2, y2) ...
       HUMAN_0_TRAJ: (x1, y1) (x2, y2) ...
       HUMAN_1_TRAJ: (x1, y1) (x2, y2) ...
    """
    episodes = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 解析episode header
        if line.startswith('[EP='):
            # Try new format first (with HUMANS field)
            match_new = re.match(
                r'\[EP=(\d+)\] START=\(([-\d.]+), ([-\d.]+)\) GOAL=\(([-\d.]+), ([-\d.]+)\) '
                r'RESULT=(\w+) STEPS=(\d+) HUMANS=(\d+)', line
            )
            # Try old format (without HUMANS field)
            match_old = re.match(
                r'\[EP=(\d+)\] START=\(([-\d.]+), ([-\d.]+)\) GOAL=\(([-\d.]+), ([-\d.]+)\) '
                r'RESULT=(\w+) STEPS=(\d+)', line
            )

            if match_new or match_old:
                if match_new:
                    ep_id = int(match_new.group(1))
                    start_x, start_y = float(match_new.group(2)), float(match_new.group(3))
                    goal_x, goal_y = float(match_new.group(4)), float(match_new.group(5))
                    result = match_new.group(6)
                    steps = int(match_new.group(7))
                    num_humans = int(match_new.group(8))
                else:
                    ep_id = int(match_old.group(1))
                    start_x, start_y = float(match_old.group(2)), float(match_old.group(3))
                    goal_x, goal_y = float(match_old.group(4)), float(match_old.group(5))
                    result = match_old.group(6)
                    steps = int(match_old.group(7))
                    num_humans = 0

                # Parse trajectories
                i += 1
                robot_traj = None
                human_trajs = []

                while i < len(lines) and lines[i].strip():
                    traj_line = lines[i].strip()

                    # Parse robot trajectory (new format: ROBOT_TRAJ:)
                    if traj_line.startswith('ROBOT_TRAJ:'):
                        coords_str = traj_line.replace('ROBOT_TRAJ:', '').strip()
                        robot_traj = parse_coords(coords_str)

                    # Parse old format (TRAJECTORY:)
                    elif traj_line.startswith('TRAJECTORY:'):
                        coords_str = traj_line.replace('TRAJECTORY:', '').strip()
                        robot_traj = parse_coords(coords_str)

                    # Parse human trajectory
                    elif traj_line.startswith('HUMAN_'):
                        match_human = re.match(r'HUMAN_(\d+)_TRAJ:\s*(.+)', traj_line)
                        if match_human:
                            human_id = int(match_human.group(1))
                            coords_str = match_human.group(2).strip()
                            human_traj = parse_coords(coords_str, allow_nan=True)
                            human_trajs.append({
                                'id': human_id,
                                'trajectory': human_traj
                            })

                    i += 1

                # Store episode data
                if robot_traj is not None:
                    episodes.append({
                        'id': ep_id,
                        'start': (start_x, start_y),
                        'goal': (goal_x, goal_y),
                        'result': result,
                        'steps': steps,
                        'trajectory': np.array(robot_traj),
                        'num_humans': num_humans,
                        'human_trajectories': human_trajs
                    })

        i += 1

    return episodes


def parse_coords(coords_str, allow_nan=False):
    """
    解析坐标字符串

    格式: (x1, y1) (x2, y2) ...
    支持nan值: (nan, nan)
    """
    points = []
    # 匹配 (x, y) 格式，包括nan
    if allow_nan:
        coords = re.findall(r'\(([-\d.nan]+), ([-\d.nan]+)\)', coords_str)
    else:
        coords = re.findall(r'\(([-\d.]+), ([-\d.]+)\)', coords_str)

    for x, y in coords:
        try:
            points.append((float(x), float(y)))
        except ValueError:
            # Handle nan values
            points.append((float('nan'), float('nan')))

    return points


def compute_metrics(episode):
    """计算单个episode的所有指标"""
    start = np.array(episode['start'])
    goal = np.array(episode['goal'])
    traj = episode['trajectory']

    if len(traj) < 2:
        return None

    # 1. 直线距离（最优路径）
    optimal_dist = np.linalg.norm(goal - start)

    # 2. 实际路径长度
    path_segments = np.diff(traj, axis=0)
    path_lengths = np.linalg.norm(path_segments, axis=1)
    actual_path_length = np.sum(path_lengths)

    # 3. 路径效率（实际/最优）
    path_efficiency = optimal_dist / actual_path_length if actual_path_length > 0 else 0

    # 4. 平滑度（角度变化）
    angles = []
    for i in range(1, len(traj) - 1):
        v1 = traj[i] - traj[i-1]
        v2 = traj[i+1] - traj[i]

        # 计算角度变化
        if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(angle)

    avg_angle_change = np.mean(angles) if angles else 0
    max_angle_change = np.max(angles) if angles else 0

    # 5. 路径偏差（距离直线的平均距离）
    # 直线方程: point = start + t * (goal - start)
    line_vec = goal - start
    line_length = np.linalg.norm(line_vec)

    if line_length > 1e-6:
        deviations = []
        for point in traj:
            # 点到直线的距离
            t = np.dot(point - start, line_vec) / (line_length ** 2)
            t = np.clip(t, 0, 1)  # 限制在起点和终点之间
            projection = start + t * line_vec
            deviation = np.linalg.norm(point - projection)
            deviations.append(deviation)

        avg_deviation = np.mean(deviations)
        max_deviation = np.max(deviations)
    else:
        avg_deviation = 0
        max_deviation = 0

    # 6. 时间效率（假设dt=0.25s）
    dt = 0.25
    actual_time = episode['steps'] * dt

    # 理论最短时间（假设v_pref=1.0 m/s）
    v_pref = 1.0
    optimal_time = optimal_dist / v_pref
    time_efficiency = optimal_time / actual_time if actual_time > 0 else 0

    # 7. 平均速度
    avg_speed = actual_path_length / actual_time if actual_time > 0 else 0

    return {
        'optimal_dist': optimal_dist,
        'actual_path_length': actual_path_length,
        'path_efficiency': path_efficiency,
        'avg_angle_change': np.degrees(avg_angle_change),  # 转换为度
        'max_angle_change': np.degrees(max_angle_change),
        'avg_deviation': avg_deviation,
        'max_deviation': max_deviation,
        'actual_time': actual_time,
        'optimal_time': optimal_time,
        'time_efficiency': time_efficiency,
        'avg_speed': avg_speed,
        'result': episode['result']
    }


def analyze_trajectories(log_path, top_n=None, plot=False):
    """主分析函数"""
    print(f"📂 读取轨迹日志: {log_path}")
    episodes = parse_trajectory_log(log_path)

    if top_n:
        episodes = episodes[:top_n]
        print(f"   分析前 {top_n} 条轨迹")

    print(f"   找到 {len(episodes)} 条轨迹\n")

    if not episodes:
        print("❌ 没有找到有效轨迹数据")
        return

    # 计算所有指标
    all_metrics = []
    for ep in episodes:
        metrics = compute_metrics(ep)
        if metrics:
            all_metrics.append(metrics)

    # 按结果分类
    success_metrics = [m for m in all_metrics if m['result'] == 'SUCCESS']
    timeout_metrics = [m for m in all_metrics if m['result'] == 'TIMEOUT']
    collision_metrics = [m for m in all_metrics if m['result'] == 'COLLISION']

    # 统计输出
    print("=" * 80)
    print("📊 轨迹质量分析报告")
    print("=" * 80)

    # 基本统计
    total = len(all_metrics)
    success_rate = len(success_metrics) / total if total > 0 else 0
    timeout_rate = len(timeout_metrics) / total if total > 0 else 0
    collision_rate = len(collision_metrics) / total if total > 0 else 0

    print(f"\n🎯 基本统计:")
    print(f"   总轨迹数:    {total}")
    print(f"   成功:        {len(success_metrics)} ({success_rate:.1%})")
    print(f"   超时:        {len(timeout_metrics)} ({timeout_rate:.1%})")
    print(f"   碰撞:        {len(collision_metrics)} ({collision_rate:.1%})")

    # 只分析成功的轨迹
    if success_metrics:
        print(f"\n📏 路径质量 (仅成功轨迹, N={len(success_metrics)}):")

        path_effs = [m['path_efficiency'] for m in success_metrics]
        print(f"   路径效率 (optimal/actual):")
        print(f"      平均: {np.mean(path_effs):.3f}  (1.0为最优, >0.85为优秀)")
        print(f"      中位: {np.median(path_effs):.3f}")
        print(f"      最佳: {np.max(path_effs):.3f}")
        print(f"      最差: {np.min(path_effs):.3f}")

        optimal_dists = [m['optimal_dist'] for m in success_metrics]
        actual_lengths = [m['actual_path_length'] for m in success_metrics]
        print(f"\n   路径长度:")
        print(f"      平均最优距离: {np.mean(optimal_dists):.2f}m")
        print(f"      平均实际路径: {np.mean(actual_lengths):.2f}m")
        print(f"      平均多走:      {np.mean(actual_lengths) - np.mean(optimal_dists):.2f}m ({(1/np.mean(path_effs) - 1)*100:.1f}%)")

        print(f"\n🎢 平滑度:")
        avg_angles = [m['avg_angle_change'] for m in success_metrics]
        max_angles = [m['max_angle_change'] for m in success_metrics]
        print(f"   平均转向角度: {np.mean(avg_angles):.1f}° (越小越平滑)")
        print(f"   最大转向角度: {np.mean(max_angles):.1f}° (急转弯指标)")

        # 分级评估
        smooth_count = sum(1 for a in avg_angles if a < 15)
        print(f"   平滑轨迹 (<15°): {smooth_count}/{len(success_metrics)} ({smooth_count/len(success_metrics):.1%})")

        print(f"\n📐 路径偏差 (距离直线):")
        avg_devs = [m['avg_deviation'] for m in success_metrics]
        max_devs = [m['max_deviation'] for m in success_metrics]
        print(f"   平均偏离: {np.mean(avg_devs):.3f}m (越小越接近直线)")
        print(f"   最大偏离: {np.mean(max_devs):.3f}m")

        # 偏差分级
        tight_count = sum(1 for d in avg_devs if d < 0.5)
        print(f"   紧凑路径 (<0.5m): {tight_count}/{len(success_metrics)} ({tight_count/len(success_metrics):.1%})")

        print(f"\n⏱️ 时间效率:")
        time_effs = [m['time_efficiency'] for m in success_metrics]
        actual_times = [m['actual_time'] for m in success_metrics]
        optimal_times = [m['optimal_time'] for m in success_metrics]
        print(f"   时间效率 (optimal/actual):")
        print(f"      平均: {np.mean(time_effs):.3f}")
        print(f"      中位: {np.median(time_effs):.3f}")
        print(f"   平均完成时间: {np.mean(actual_times):.2f}s (理论最优: {np.mean(optimal_times):.2f}s)")

        print(f"\n🚀 速度分析:")
        speeds = [m['avg_speed'] for m in success_metrics]
        print(f"   平均速度: {np.mean(speeds):.3f}m/s (v_pref=1.0m/s)")
        print(f"   速度利用率: {np.mean(speeds)/1.0:.1%}")

        print(f"\n⭐ 综合质量评估:")
        # 综合评分：路径效率40% + 平滑度30% + 时间效率30%
        quality_scores = []
        for m in success_metrics:
            path_score = m['path_efficiency'] * 40
            smooth_score = max(0, (30 - m['avg_angle_change']) / 30 * 30)  # 30度为0分
            time_score = m['time_efficiency'] * 30
            quality = path_score + smooth_score + time_score
            quality_scores.append(quality)

        print(f"   平均质量得分: {np.mean(quality_scores):.1f}/100")

        excellent = sum(1 for s in quality_scores if s >= 80)
        good = sum(1 for s in quality_scores if 60 <= s < 80)
        fair = sum(1 for s in quality_scores if 40 <= s < 60)
        poor = sum(1 for s in quality_scores if s < 40)

        print(f"   优秀 (≥80分): {excellent} ({excellent/len(success_metrics):.1%})")
        print(f"   良好 (60-80): {good} ({good/len(success_metrics):.1%})")
        print(f"   一般 (40-60): {fair} ({fair/len(success_metrics):.1%})")
        print(f"   较差 (<40):   {poor} ({poor/len(success_metrics):.1%})")

    # 超时分析
    if timeout_metrics:
        print(f"\n⏳ 超时轨迹分析 (N={len(timeout_metrics)}):")
        timeout_lengths = [m['actual_path_length'] for m in timeout_metrics]
        timeout_times = [m['actual_time'] for m in timeout_metrics]
        print(f"   平均路径长度: {np.mean(timeout_lengths):.2f}m")
        print(f"   平均耗时: {np.mean(timeout_times):.2f}s")
        print(f"   平均速度: {np.mean([m['avg_speed'] for m in timeout_metrics]):.3f}m/s")

    print("\n" + "=" * 80)

    # 可视化
    if plot and success_metrics:
        plot_analysis(success_metrics, log_path)
        # 生成轨迹偏差"微积分图"（选择最好、平均、最差3条）
        plot_deviation_integral(episodes, success_metrics, log_path)
        # 生成机器人+行人避障场景图
        plot_obstacle_avoidance_scenes(episodes, log_path)


def plot_analysis(metrics, log_path):
    """生成可视化图表"""
    # 设置全局字体大小（避免图例文字遮挡图表）
    plt.rcParams.update({
        'font.size': 8,           # 基础字体大小
        'axes.titlesize': 10,     # 子图标题
        'axes.labelsize': 9,      # 轴标签
        'xtick.labelsize': 8,     # x轴刻度
        'ytick.labelsize': 8,     # y轴刻度
        'legend.fontsize': 7,     # 图例字体（关键：减小图例文字）
    })

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Trajectory Quality Analysis', fontsize=13, fontweight='bold')

    # 1. 路径效率分布
    ax = axes[0, 0]
    path_effs = [m['path_efficiency'] for m in metrics]
    ax.hist(path_effs, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(path_effs), color='r', linestyle='--', label=f'Mean: {np.mean(path_effs):.3f}')
    ax.axvline(0.85, color='g', linestyle='--', label='Excellent (0.85)')
    ax.set_xlabel('Path Efficiency (optimal/actual)')
    ax.set_ylabel('Frequency')
    ax.set_title('Path Efficiency Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. 平滑度分布
    ax = axes[0, 1]
    avg_angles = [m['avg_angle_change'] for m in metrics]
    ax.hist(avg_angles, bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(np.mean(avg_angles), color='r', linestyle='--', label=f'Mean: {np.mean(avg_angles):.1f}°')
    ax.axvline(15, color='g', linestyle='--', label='Smooth (<15°)')
    ax.set_xlabel('Average Angle Change (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Path Smoothness Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. 路径偏差分布
    ax = axes[0, 2]
    avg_devs = [m['avg_deviation'] for m in metrics]
    ax.hist(avg_devs, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(np.mean(avg_devs), color='r', linestyle='--', label=f'Mean: {np.mean(avg_devs):.3f}m')
    ax.axvline(0.5, color='g', linestyle='--', label='Tight (<0.5m)')
    ax.set_xlabel('Average Deviation from Straight Line (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Path Deviation Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. 时间效率分布
    ax = axes[1, 0]
    time_effs = [m['time_efficiency'] for m in metrics]
    ax.hist(time_effs, bins=20, edgecolor='black', alpha=0.7, color='purple')
    ax.axvline(np.mean(time_effs), color='r', linestyle='--', label=f'Mean: {np.mean(time_effs):.3f}')
    ax.set_xlabel('Time Efficiency (optimal/actual)')
    ax.set_ylabel('Frequency')
    ax.set_title('Time Efficiency Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    # 5. 路径长度 vs 最优距离
    ax = axes[1, 1]
    optimal_dists = [m['optimal_dist'] for m in metrics]
    actual_lengths = [m['actual_path_length'] for m in metrics]
    ax.scatter(optimal_dists, actual_lengths, alpha=0.5)

    # 添加y=x参考线（完美路径）
    max_val = max(max(optimal_dists), max(actual_lengths))
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect Path (y=x)')

    # 添加回归线
    z = np.polyfit(optimal_dists, actual_lengths, 1)
    p = np.poly1d(z)
    ax.plot(optimal_dists, p(optimal_dists), 'g-', label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

    ax.set_xlabel('Optimal Distance (m)')
    ax.set_ylabel('Actual Path Length (m)')
    ax.set_title('Actual vs Optimal Path Length')
    ax.legend()
    ax.grid(alpha=0.3)

    # 6. 综合质量得分分布
    ax = axes[1, 2]
    quality_scores = []
    for m in metrics:
        path_score = m['path_efficiency'] * 40
        smooth_score = max(0, (30 - m['avg_angle_change']) / 30 * 30)
        time_score = m['time_efficiency'] * 30
        quality = path_score + smooth_score + time_score
        quality_scores.append(quality)

    ax.hist(quality_scores, bins=20, edgecolor='black', alpha=0.7, color='cyan')
    ax.axvline(np.mean(quality_scores), color='r', linestyle='--', label=f'Mean: {np.mean(quality_scores):.1f}')
    ax.axvline(80, color='g', linestyle='--', label='Excellent (80)')
    ax.axvline(60, color='orange', linestyle='--', label='Good (60)')
    ax.set_xlabel('Quality Score (0-100)')
    ax.set_ylabel('Frequency')
    ax.set_title('Overall Quality Score Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # 保存图表
    output_path = Path(log_path).parent / 'trajectory_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 可视化图表已保存: {output_path}")

    plt.close()


def classify_trajectory_type(deviations):
    """
    分类轨迹类型：单面拐弯 vs S形拐弯

    返回：
    - 'single': 单面拐弯（偏差始终同号）
    - 's_shape': S形拐弯（偏差变号≥1次）
    - sign_changes: 变号次数
    """
    signs = np.sign(deviations)
    # 统计符号变化次数
    sign_changes = np.sum(np.abs(np.diff(signs[signs != 0])) > 0)

    if sign_changes <= 1:
        return 'single', sign_changes
    else:
        return 's_shape', sign_changes


def plot_deviation_integral(episodes, metrics, log_path):
    """
    绘制轨迹偏差的"微积分图"（类似黎曼和）
    展示6条代表性轨迹：3条单面拐弯 + 3条S形拐弯

    选择逻辑：
    1. 计算所有轨迹的偏差数据
    2. 分类为"单面"和"S形"两组
    3. 每组按路径效率排序，选择Best/Median/Worst
    """
    # 先计算所有轨迹的偏差数据和类型
    success_episodes = [ep for ep in episodes if ep['result'] == 'SUCCESS']

    trajectory_data = []
    for idx, (episode, metric) in enumerate(zip(success_episodes, metrics)):
        start = np.array(episode['start'])
        goal = np.array(episode['goal'])
        traj = episode['trajectory']

        line_vec = goal - start
        line_length = np.linalg.norm(line_vec)

        # 计算偏差
        deviations = []
        for point in traj:
            t = np.dot(point - start, line_vec) / (line_length ** 2) if line_length > 1e-6 else 0
            t = np.clip(t, 0, 1)
            projection = start + t * line_vec

            # 带符号的偏差
            cross = (point[0] - start[0]) * (goal[1] - start[1]) - (point[1] - start[1]) * (goal[0] - start[0])
            deviation = np.linalg.norm(point - projection)
            signed_deviation = deviation if cross > 0 else -deviation
            deviations.append(signed_deviation)

        # 分类轨迹类型
        traj_type, sign_changes = classify_trajectory_type(deviations)

        trajectory_data.append({
            'idx': idx,
            'episode': episode,
            'metric': metric,
            'deviations': deviations,
            'type': traj_type,
            'sign_changes': sign_changes,
            'efficiency': metric['path_efficiency']
        })

    # 按类型分组
    single_trajs = [t for t in trajectory_data if t['type'] == 'single']
    s_shape_trajs = [t for t in trajectory_data if t['type'] == 's_shape']

    print(f"\n📊 轨迹分类统计:")
    print(f"   单面拐弯: {len(single_trajs)} ({len(single_trajs)/len(trajectory_data)*100:.1f}%)")
    print(f"   S形拐弯: {len(s_shape_trajs)} ({len(s_shape_trajs)/len(trajectory_data)*100:.1f}%)")

    # 如果S形轨迹太少，降级为3+3或3+2
    if len(s_shape_trajs) < 3:
        print(f"   ⚠️ S形轨迹不足3条，将展示所有可用S形轨迹")

    # 每组选择Best/Median/Worst
    def select_trajectories(traj_list, count=3):
        if len(traj_list) == 0:
            return []
        sorted_list = sorted(traj_list, key=lambda x: x['efficiency'], reverse=True)
        if len(sorted_list) < count:
            return sorted_list
        indices = [0, len(sorted_list)//2, len(sorted_list)-1]
        return [sorted_list[i] for i in indices]

    selected_single = select_trajectories(single_trajs, 3)
    selected_s_shape = select_trajectories(s_shape_trajs, 3)

    selected_trajs = selected_single + selected_s_shape

    # 创建3×2布局（左边3个单面，右边3个S形）
    n_plots = len(selected_trajs)

    # 设置字体大小（与主图表一致）
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
    })

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    axes = axes.flatten()  # 转成1维数组便于索引

    fig.suptitle('Trajectory Deviation "Integral" View: Single-Side (Left) vs S-Shape (Right)',
                 fontsize=13, fontweight='bold')

    # 如果轨迹不足6条，隐藏多余的子图
    for i in range(n_plots, 6):
        axes[i].axis('off')

    # 绘制每条轨迹
    for plot_idx, traj_data in enumerate(selected_trajs):
        ax = axes[plot_idx]

        episode = traj_data['episode']
        metric = traj_data['metric']
        deviations = traj_data['deviations']
        traj_type = traj_data['type']
        sign_changes = traj_data['sign_changes']

        start = np.array(episode['start'])
        goal = np.array(episode['goal'])
        traj = episode['trajectory']
        line_vec = goal - start
        line_length = np.linalg.norm(line_vec)

        # 计算沿直线的距离
        distances_along_line = []
        for point in traj:
            t = np.dot(point - start, line_vec) / (line_length ** 2) if line_length > 1e-6 else 0
            t = np.clip(t, 0, 1)
            distances_along_line.append(t * line_length)

        # 确定质量标签（Best/Median/Worst）
        if plot_idx < len(selected_single):
            category = "Single-Side"
            if plot_idx == 0:
                quality_label = "BEST"
                quality_color = "green"
            elif plot_idx == len(selected_single) - 1:
                quality_label = "WORST"
                quality_color = "red"
            else:
                quality_label = "MEDIAN"
                quality_color = "orange"
        else:
            category = "S-Shape"
            rel_idx = plot_idx - len(selected_single)
            if rel_idx == 0:
                quality_label = "BEST"
                quality_color = "green"
            elif rel_idx == len(selected_s_shape) - 1:
                quality_label = "WORST"
                quality_color = "red"
            else:
                quality_label = "MEDIAN"
                quality_color = "orange"

        # 绘制基准线（y=0）
        ax.axhline(0, color='red', linewidth=2, linestyle='--', label='Optimal Path', zorder=1)

        # 创建平滑曲线（使用样条插值）
        if len(distances_along_line) >= 4:
            distances_smooth = np.linspace(min(distances_along_line), max(distances_along_line), 300)
            try:
                spl = make_interp_spline(distances_along_line, deviations, k=3)
                deviations_smooth = spl(distances_smooth)
            except:
                deviations_smooth = np.interp(distances_smooth, distances_along_line, deviations)
            ax.plot(distances_smooth, deviations_smooth, 'b-', linewidth=2, label='Actual Deviation', zorder=2)
            ax.fill_between(distances_smooth, 0, deviations_smooth,
                           alpha=0.2, color='blue', label='Cumulative Area')
        else:
            ax.plot(distances_along_line, deviations, 'b-', linewidth=2, label='Actual Deviation', zorder=2)
            ax.fill_between(distances_along_line, 0, deviations,
                           alpha=0.2, color='blue', label='Cumulative Area')

        # 绘制"竖线"（黎曼和效果）- 加深颜色
        for i, (dist, dev) in enumerate(zip(distances_along_line, deviations)):
            # 使用更深的颜色和更高的不透明度
            if abs(dev) < 0.5:
                color = 'darkgreen'
                alpha = 0.5
            elif abs(dev) < 1.0:
                color = 'darkorange'
                alpha = 0.6
            else:
                color = 'darkred'
                alpha = 0.7
            ax.plot([dist, dist], [0, dev], color=color, linewidth=1.0, alpha=alpha, zorder=0)

        # 添加质量标签（醒目的文字）
        ax.text(0.98, 0.98, quality_label, transform=ax.transAxes,
               fontsize=20, fontweight='bold', color=quality_color,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=quality_color, linewidth=3))

        # 标题
        ax.set_title(f'[{category}] Efficiency: {metric["path_efficiency"]:.3f} | '
                    f'Avg Deviation: {metric["avg_deviation"]:.3f}m | '
                    f'Sign Changes: {sign_changes}x | '
                    f'Path: {metric["actual_path_length"]:.2f}m vs {metric["optimal_dist"]:.2f}m',
                    fontsize=11, pad=8)
        ax.set_xlabel('Distance Along Optimal Path (m)', fontsize=10)
        ax.set_ylabel('Signed Deviation (m)', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)

        # 统计信息文本框
        stats_text = (
            f'Area: {np.sum(np.abs(deviations)) * (line_length / len(deviations)):.2f} m²\n'
            f'Max Dev: {np.max(np.abs(deviations)):.3f}m\n'
            f'Steps: {len(traj)}'
        )
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    plt.tight_layout()

    # 保存图表
    output_path = Path(log_path).parent / 'trajectory_deviation_integral.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 轨迹偏差'微积分图'已保存: {output_path}")

    plt.close()


def plot_obstacle_avoidance_scenes(episodes, log_path):
    """
    绘制机器人+行人避障场景图

    选择6个代表性episode（按路径效率排序：Best/Good/Median/Fair/Poor/Worst）
    每个episode显示：
    - 机器人轨迹（蓝色实线）
    - 行人轨迹（红色虚线）
    - 起点/终点标记
    - 碰撞检测圈（半径0.3m）
    """
    # 只选择成功的episodes
    success_episodes = [ep for ep in episodes if ep['result'] == 'SUCCESS' and ep.get('num_humans', 0) > 0]

    if not success_episodes:
        print(f"\n⚠️ 没有找到包含行人的成功轨迹，跳过避障场景图生成")
        return

    # 计算路径效率并排序
    episodes_with_eff = []
    for ep in success_episodes:
        start = np.array(ep['start'])
        goal = np.array(ep['goal'])
        traj = ep['trajectory']

        optimal_dist = np.linalg.norm(goal - start)
        path_segments = np.diff(traj, axis=0)
        actual_path_length = np.sum(np.linalg.norm(path_segments, axis=1))
        path_efficiency = optimal_dist / actual_path_length if actual_path_length > 0 else 0

        episodes_with_eff.append({
            'episode': ep,
            'efficiency': path_efficiency
        })

    # 按效率排序
    episodes_with_eff.sort(key=lambda x: x['efficiency'], reverse=True)

    # 选择6个代表性episode
    n = len(episodes_with_eff)
    if n >= 6:
        selected_indices = [0, n//5, n//2, 3*n//5, 4*n//5, n-1]
        labels = ["BEST", "GOOD", "MEDIAN", "FAIR", "POOR", "WORST"]
        colors = ["darkgreen", "green", "orange", "darkorange", "red", "darkred"]
    elif n >= 3:
        selected_indices = [0, n//2, n-1]
        labels = ["BEST", "MEDIAN", "WORST"]
        colors = ["darkgreen", "orange", "darkred"]
    else:
        selected_indices = list(range(n))
        labels = ["BEST"] * n
        colors = ["darkgreen"] * n

    selected = [episodes_with_eff[i] for i in selected_indices]

    # 创建3×2布局
    n_plots = len(selected)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    fig.suptitle('Robot-Human Obstacle Avoidance Scenarios', fontsize=14, fontweight='bold')

    # 隐藏多余的子图
    for i in range(n_plots, 6):
        axes[i].axis('off')

    # 绘制每个场景
    for plot_idx, data in enumerate(selected):
        ax = axes[plot_idx]
        ep = data['episode']
        efficiency = data['efficiency']

        robot_traj = ep['trajectory']
        start = np.array(ep['start'])
        goal = np.array(ep['goal'])

        # 绘制机器人轨迹
        ax.plot(robot_traj[:, 0], robot_traj[:, 1], 'b-', linewidth=2.5, label='Robot Path', zorder=3)

        # 绘制起点和终点
        ax.scatter(start[0], start[1], c='green', s=200, marker='o', edgecolors='black', linewidths=2, label='Start', zorder=5)
        ax.scatter(goal[0], goal[1], c='red', s=200, marker='*', edgecolors='black', linewidths=2, label='Goal', zorder=5)

        # 绘制机器人位置圈（每5步）
        robot_radius = 0.3
        for i in range(0, len(robot_traj), 5):
            circle = plt.Circle((robot_traj[i, 0], robot_traj[i, 1]), robot_radius,
                              color='blue', alpha=0.1, zorder=1)
            ax.add_patch(circle)

        # 绘制行人轨迹
        human_trajs = ep.get('human_trajectories', [])
        human_colors = ['red', 'orange', 'purple', 'brown', 'pink']

        for h_idx, human_data in enumerate(human_trajs):
            human_traj = np.array(human_data['trajectory'])
            # 过滤掉nan值
            valid_mask = ~np.isnan(human_traj[:, 0])
            if not valid_mask.any():
                continue

            human_traj_valid = human_traj[valid_mask]
            h_color = human_colors[h_idx % len(human_colors)]

            # 绘制行人轨迹
            ax.plot(human_traj_valid[:, 0], human_traj_valid[:, 1], '--',
                   color=h_color, linewidth=2, alpha=0.7, label=f'Human {human_data["id"]}', zorder=2)

            # 绘制行人位置圈（每5步）
            human_radius = 0.3
            for i in range(0, len(human_traj_valid), 5):
                circle = plt.Circle((human_traj_valid[i, 0], human_traj_valid[i, 1]), human_radius,
                                  color=h_color, alpha=0.15, zorder=0)
                ax.add_patch(circle)

        # 绘制最优路径（虚线）
        ax.plot([start[0], goal[0]], [start[1], goal[1]], 'k--',
               linewidth=1.5, alpha=0.3, label='Optimal Path', zorder=0)

        # 添加质量标签
        quality_label = labels[plot_idx]
        quality_color = colors[plot_idx]
        ax.text(0.98, 0.98, quality_label, transform=ax.transAxes,
               fontsize=18, fontweight='bold', color=quality_color,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor=quality_color, linewidth=3))

        # 标题
        ax.set_title(f'[EP {ep["id"]}] Efficiency: {efficiency:.3f} | Humans: {ep["num_humans"]} | Steps: {ep["steps"]}',
                    fontsize=11, pad=8)
        ax.set_xlabel('X Position (m)', fontsize=10)
        ax.set_ylabel('Y Position (m)', fontsize=10)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # 保存图表
    output_path = Path(log_path).parent / 'obstacle_avoidance_scenes.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 避障场景图已保存: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze navigation trajectory quality')
    parser.add_argument('--log', type=str, required=True, help='Path to trajectory.log')
    parser.add_argument('--top', type=int, default=None, help='Analyze only top N trajectories')
    parser.add_argument('--plot', action='store_true', help='Generate visualization plots')

    args = parser.parse_args()

    if not Path(args.log).exists():
        print(f"❌ 文件不存在: {args.log}")
        return

    analyze_trajectories(args.log, top_n=args.top, plot=args.plot)


if __name__ == '__main__':
    main()
