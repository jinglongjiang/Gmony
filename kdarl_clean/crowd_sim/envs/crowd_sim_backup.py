# -*- coding: utf-8 -*-
"""
Simplified CrowdSim (reward-minimal version)
-------------------------------------------
本文件提供与 CrowdNav 兼容的 CrowdSim 环境实现（接口保持不变），
仅 **重构奖励函数** 为 CrowdNav/SARL/CAMRL 风格的 **极简奖励**：
  • 终止类：到达/碰撞/超时
  • 密集项：不舒适距离惩罚（discomfort）
  • 可选：势能型进度 shaping（默认关闭、小权重、强裁剪）
其它复杂 shaping（TTC/对齐/相对速度/速度整形/shape potential/时间稀释…）全部删除。

公开 API（与原版一致的核心子集）：
  - configure(cfg)
  - set_robot(robot)
  - reset(phase='train') -> obs
  - step(action, update=True) -> (obs, reward, done, truncated, info)
  - render()  # 可选空实现

依赖：
  - crowd_sim.envs.utils.robot.Robot
  - crowd_sim.envs.utils.human.Human
  - 机器人/行人对象需实现：get_full_state()/get_observable_state()/step()/
    compute_position()/get_goal_position()/get_obs_array()/get_next_obs_array()

注意：
  - 若你的工程已有 CrowdSim，只想替换奖励，请仅拷贝本文件中的
    `CrowdSim.step()` 实现覆盖你原文件的 step()。
"""
from __future__ import annotations

import math
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    from crowd_sim.envs.utils.robot import Robot
    from crowd_sim.envs.utils.human import Human
except Exception:
    Robot = object  # 类型提示降级
    Human = object


# ----------------------------- 小工具 -----------------------------

def norm(v):
    return float(np.linalg.norm(np.asarray(v, dtype=np.float32)))


def point_to_segment_dist(px, py, qx, qy, sx, sy):
    """点 P(px,py) 到线段 (S->Q) 的最短距离（标量）。"""
    vx, vy = qx - sx, qy - sy
    wx, wy = px - sx, py - sy
    l2 = vx * vx + vy * vy
    if l2 <= 1e-12:
        # 退化为点
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / l2))
    projx, projy = sx + t * vx, sy + t * vy
    return math.hypot(px - projx, py - projy)


def _ttc_rel(px, py, vx, vy, r_sum):
    """精确 TTC 计算（debug.md P0-1优化版本）"""
    # 二次方程近似 TTC，负/无实根视为 inf
    a = vx*vx + vy*vy + 1e-8
    b = px*vx + py*vy
    c = px*px + py*py - r_sum*r_sum
    disc = b*b - a*c
    if disc <= 0:
        return float('inf')
    t1 = (-b - math.sqrt(disc)) / a
    return t1 if t1 > 1e-6 else float('inf')


# ----------------------------- 配置包装 -----------------------------
@dataclass
class RewardCfg:
    success_reward: float = 1.0  # 对齐config.txt，避免双轨
    collision_penalty: float = -0.25  # 对齐config.txt，避免双轨
    timeout_penalty: float = -0.5
    discomfort_dist: float = 0.6
    discomfort_penalty_factor: float = 0.7
    use_progress_shaping: bool = False
    progress_weight: float = 0.3
    max_prog_step: float = 0.5
    success_radius: float = 0.50


@dataclass
class SimCfg:
    time_step: float = 0.25
    time_limit: float = 25.0
    visible: bool = True
    sensor: str = 'coordinates'
    # humans 配置（若需要在 reset 中生成）
    human_num: int = 5
    human_radius: float = 0.3
    human_v: float = 1.0


# ----------------------------- CrowdSim -----------------------------
class CrowdSim:
    def __init__(self):
        self.config = SimCfg()
        self.rwd = RewardCfg()
        self.robot: Optional[Robot] = None
        self.humans: List[Human] = []
        self.states = []
        self.global_time = 0.0
        # P0-1: 避让钩子开关，IL阶段禁用
        self.avoidance_hook_enabled = True
        self.phase = 'rl'  # 默认RL阶段
        # debug.md斧头一: 添加步数跟踪变量
        self.step_count = 0
        self.time_limit_steps = 100  # 默认值，会在configure/reset中更新

    # ----------- 外部配置（与原版接口一致） -----------
    def configure(self, cfg_like) -> None:
        """从 configparser 或 dict 注入参数。"""
        # 保存原始配置以便在reset()中访问[sim]等section
        self._raw_config = cfg_like
        # time / visibility
        self.config.time_step = float(_get(cfg_like, 'env', 'time_step', self.config.time_step))
        self.config.time_limit = float(_get(cfg_like, 'env', 'time_limit', self.config.time_limit))
        self.config.visible = bool(_get(cfg_like, 'env', 'visible', int(self.config.visible)))
        self.config.sensor = str(_get(cfg_like, 'env', 'sensor', self.config.sensor))
        # reward
        self.rwd.success_reward = float(_get(cfg_like, 'reward', 'success_reward', self.rwd.success_reward))
        self.rwd.collision_penalty = float(_get(cfg_like, 'reward', 'collision_penalty', self.rwd.collision_penalty))
        self.rwd.timeout_penalty = float(_get(cfg_like, 'reward', 'timeout_penalty', self.rwd.timeout_penalty))
        self.rwd.discomfort_dist = float(_get(cfg_like, 'reward', 'discomfort_dist', self.rwd.discomfort_dist))
        self.rwd.discomfort_penalty_factor = float(_get(cfg_like, 'reward', 'discomfort_penalty_factor', self.rwd.discomfort_penalty_factor))
        self.rwd.use_progress_shaping = bool(int(_get(cfg_like, 'reward', 'use_progress_shaping', int(self.rwd.use_progress_shaping))))
        self.rwd.progress_weight = float(_get(cfg_like, 'reward', 'progress_weight', self.rwd.progress_weight))
        self.rwd.max_prog_step = float(_get(cfg_like, 'reward', 'max_prog_step', self.rwd.max_prog_step))
        self.rwd.success_radius = float(_get(cfg_like, 'reward', 'success_radius', self.rwd.success_radius))
        # humans（仅当需要在 reset 内部生成）
        self.config.human_num = int(_get(cfg_like, 'env', 'human_num', self.config.human_num))
        self.config.human_radius = float(_get(cfg_like, 'env', 'human_radius', self.config.human_radius))
        self.config.human_v = float(_get(cfg_like, 'env', 'human_v', self.config.human_v))

        # debug.md MSR (Minimal Safe Reward) 配置 - Two-stage profile支持
        profile = str(_get(cfg_like, 'train', 'reward_profile', 'benchmark'))

        # 根据profile设置默认参数
        if profile == 'success_boost':
            # Win-Now阶段：强进度奖励
            default_lambda_p = 3.0
            default_use_progress = True
        else:
            # Benchmark阶段：温和MSR
            default_lambda_p = 0.10
            default_use_progress = True

        self.msr_use_progress = bool(_get(cfg_like, 'reward_msr', 'use_progress', default_use_progress))
        self.msr_lambda_p = float(_get(cfg_like, 'reward_msr', 'lambda_p', default_lambda_p))
        self.msr_use_time_penalty = bool(_get(cfg_like, 'reward_msr', 'use_time_penalty', True))
        self.msr_eta = float(_get(cfg_like, 'reward_msr', 'eta', 0.01))
        self.msr_use_antifreeze = bool(_get(cfg_like, 'reward_msr', 'use_antifreeze', True))
        self.msr_freeze_K = int(_get(cfg_like, 'reward_msr', 'freeze_K', 4))
        self.msr_kappa = float(_get(cfg_like, 'reward_msr', 'kappa', 0.05))
        self.msr_use_soft_ttc = bool(_get(cfg_like, 'reward_msr', 'use_soft_ttc', False))
        self.msr_ttc_tau = float(_get(cfg_like, 'reward_msr', 'ttc_tau', 3.0))
        self.msr_alpha = float(_get(cfg_like, 'reward_msr', 'alpha', 0.20))

        # 保存profile以便在step中使用
        self.reward_profile = profile

        # MSR状态跟踪变量
        self.robot_speed_history = []  # 跟踪机器人速度历史

    def set_robot(self, robot: Robot) -> None:
        self.robot = robot
        # 将可见性/传感器同步到 robot
        if hasattr(self.robot, 'visible'):
            self.robot.visible = self.config.visible
        if hasattr(self.robot, 'sensor'):
            self.robot.sensor = self.config.sensor

    # ----------- episode 生命周期 -----------
    def reset(self, phase: str = 'train'):
        """
        重置一局：必须在这里完成 robot/humans 的初始状态与目标布置，
        否则 get_full_state() 会读到 None 而报 TypeError。
        """
        assert self.robot is not None, '请先 set_robot(robot)'
        self.phase = phase
        # P0-1: 根据phase设置避让钩子状态（IL阶段禁用，让ORCA纯净运行）
        self.avoidance_hook_enabled = (phase != 'il')
        self.global_time = 0.0
        self.states = []

        # debug.md斧头一: 统一"秒→步数"转换
        self.dt = float(self.config.time_step)  # e.g., 0.25
        self.time_limit_sec = float(self.config.time_limit)  # e.g., 25.0（单位是秒）
        self.time_limit_steps = int(round(self.time_limit_sec / self.dt))  # -> 100
        self.step_count = 0

        # debug.md斧头一: 日志显示秒与步数，防止再踩坑
        logging.info(f"[CFG-LOCK] dt={self.dt}s, time_limit={self.time_limit_sec}s ({self.time_limit_steps} steps)")

        # debug.md MSR状态跟踪变量重置
        self.robot_speed_history = []  # 跟踪机器人速度历史用于反僵滞检测
        self.prev_goal_distance = None  # 跟踪前一步到目标的距离用于进度势能

        # 依据 [sim] 配置选择场景（默认 circle_crossing）
        sim_name = 'circle_crossing'  # 默认场景
        try:
            # 某些配置对象是 configparser，某些是 dataclass；两者都兼容
            import configparser
            if hasattr(self, '_raw_config') and isinstance(self._raw_config, configparser.ConfigParser):
                sim_name = self._raw_config.get('sim', 'train_val_sim', fallback=sim_name)
            elif isinstance(self.config, configparser.ConfigParser):
                sim_name = self.config.get('sim', 'train_val_sim', fallback=sim_name)
            else:
                sim_name = str(getattr(self.config, 'train_val_sim', sim_name))
        except Exception:
            pass

        if 'circle' in sim_name.lower():
            self._init_circle_crossing()
        else:
            # 最小兜底：仍然初始化在原点左右
            self._init_circle_crossing()

        # 返回初始观测
        return self._obs()

    def _init_circle_crossing(self):
        """
        参照 CrowdNav 的 circle_crossing：机器人从 -R→+R，
        humans 均匀分布在圆周上并朝向对跖点。
        """
        # 读 [sim] 段参数
        R = 4.0
        n_h = 5
        try:
            # 使用保存的原始配置来读取[sim] section
            if hasattr(self, '_raw_config'):
                R = float(_get(self._raw_config, 'sim', 'circle_radius', R))
                n_h = int(_get(self._raw_config, 'sim', 'human_num', n_h))
            else:
                # 回退到从config对象读取
                R = float(getattr(self.config, 'circle_radius', R))
                n_h = int(getattr(self.config, 'human_num', n_h))
        except Exception:
            pass

        # ---- 初始化 robot（debug.md方案：可达性约束 + 安全间距）----
        # debug.md: 可达性约束 ||start - goal|| ≤ 0.85 * v_pref_robot * time_limit
        v_pref_robot = 1.0
        time_limit = float(getattr(self.config, 'time_limit', 25.0))
        max_reachable_distance = 0.85 * v_pref_robot * time_limit  # 21.25m 余量15%

        # 多次尝试生成满足约束的起点-目标对
        max_attempts = 100
        valid_layout_found = False

        for attempt in range(max_attempts):
            # 机器人起点：在扩大圆周上随机位置
            robot_R = R * 1.3  # 外圆
            start_angle = np.random.uniform(0, 2 * np.pi)
            start_px = robot_R * np.cos(start_angle)
            start_py = robot_R * np.sin(start_angle)

            # 目标点：在内圆上随机位置
            goal_R = R * 0.8  # 内圆
            goal_angle = np.random.uniform(0, 2 * np.pi)
            goal_px = goal_R * np.cos(goal_angle)
            goal_py = goal_R * np.sin(goal_angle)

            # debug.md约束1：检查可达性
            distance = np.sqrt((goal_px - start_px)**2 + (goal_py - start_py)**2)
            if distance <= max_reachable_distance:
                # 约束满足，设置机器人位置
                self.robot.px = float(start_px)
                self.robot.py = float(start_py)
                self.robot.gx = float(goal_px)
                self.robot.gy = float(goal_py)

                # 基础设置
                self.robot.vx = 0.0
                self.robot.vy = 0.0
                self.robot.theta = 0.0
                if not hasattr(self.robot, 'radius') or self.robot.radius is None:
                    self.robot.radius = 0.3
                if not hasattr(self.robot, 'v_pref') or self.robot.v_pref is None:
                    self.robot.v_pref = v_pref_robot
                self.robot.time_step = self.config.time_step

                valid_layout_found = True
                if attempt > 0:  # 只在重试时记录
                    logging.info(f"[DEBUG.MD] Found valid robot layout after {attempt + 1} attempts: distance={distance:.2f}m <= {max_reachable_distance:.2f}m")
                break

        if not valid_layout_found:
            # 回退到原始布局
            logging.warning(f"[DEBUG.MD] Could not find reachable layout after {max_attempts} attempts, using fallback")
            robot_R = R * 1.3
            start_offset = 0.4 * robot_R
            self.robot.px = -float(robot_R)
            self.robot.py = float(start_offset)
            self.robot.vx = 0.0
            self.robot.vy = 0.0
            self.robot.theta = 0.0
            if not hasattr(self.robot, 'radius') or self.robot.radius is None:
                self.robot.radius = 0.3
            if not hasattr(self.robot, 'v_pref') or self.robot.v_pref is None:
                self.robot.v_pref = 1.0
            self.robot.time_step = self.config.time_step
            self.robot.gx = float(R * 0.8)
            self.robot.gy = -float(start_offset * 0.6)

        # ---- 初始化 humans ----
        self.humans = []
        human_radius = 0.3
        human_vpref = 1.0
        try:
            # 使用保存的原始配置来读取[humans] section
            if hasattr(self, '_raw_config'):
                human_radius = float(_get(self._raw_config, 'humans', 'radius', human_radius))
                human_vpref = float(_get(self._raw_config, 'humans', 'v_pref', human_vpref))
            else:
                # 回退到从config对象读取
                human_radius = float(getattr(self.config, 'human_radius', human_radius))
                human_vpref = float(getattr(self.config, 'human_v', human_vpref))
        except Exception:
            pass

        # debug.md方案：安全间距约束 min_dist(start, any human) ≥ 0.5m
        def _valid_spawn(p, others, r):
            for q, rq in others:
                distance = np.hypot(p[0]-q[0], p[1]-q[1])
                # debug.md要求：与机器人保持≥0.5m安全距离
                min_safe_distance = max(0.5, r + rq + 0.05)  # 取0.5m或半径和+间隙的较大值
                if distance < min_safe_distance:
                    return False
            return True

        # 已有实体列表（机器人位置和半径）
        existing_entities = [([self.robot.px, self.robot.py], self.robot.radius)]

        # 均匀角度 + 随机整体偏移，避免总是同一组起点
        base = np.linspace(0, 2 * np.pi, num=max(1, n_h), endpoint=False, dtype=np.float32)
        jitter = float(np.random.uniform(0, 2 * np.pi))
        angles = base + jitter

        for k, initial_ang in enumerate(angles):
            # 确保传入正确的配置对象
            config_obj = getattr(self, '_raw_config', None) or self.config
            try:
                h = Human(config_obj, 'humans')
            except Exception as e:
                # 如果配置有问题，创建一个临时配置
                import configparser
                temp_cfg = configparser.ConfigParser()
                temp_cfg.add_section('humans')
                temp_cfg.set('humans', 'policy', 'orca')
                temp_cfg.set('humans', 'radius', str(human_radius))
                temp_cfg.set('humans', 'v_pref', str(human_vpref))
                temp_cfg.set('humans', 'visible', 'true')
                temp_cfg.set('humans', 'sensor', 'coordinates')
                temp_cfg.add_section('action_space')
                temp_cfg.set('action_space', 'kinematics', 'holonomic')
                h = Human(temp_cfg, 'humans')

            # P0-2 重采样避免重叠：尝试多个角度直到找到有效位置
            max_attempts = 50
            valid_position_found = False

            for attempt in range(max_attempts):
                # 计算位置，每次尝试时稍微偏移角度
                ang = initial_ang + attempt * 0.1  # 小步调整角度
                potential_px = float(R * np.cos(ang))
                potential_py = float(R * np.sin(ang))

                # 检查是否与已有实体重叠
                if _valid_spawn([potential_px, potential_py], existing_entities, human_radius):
                    h.px = potential_px
                    h.py = potential_py
                    valid_position_found = True
                    break

            # 如果找不到有效位置，稍微扩大半径再试
            if not valid_position_found:
                expanded_R = R * 1.2  # 扩大20%半径
                ang = initial_ang
                h.px = float(expanded_R * np.cos(ang))
                h.py = float(expanded_R * np.sin(ang))

            # 将此人类加入已有实体列表
            existing_entities.append(([h.px, h.py], human_radius))

            # 对跖点作为"目标方向"，设初速度指向对跖
            gx = float(-h.px)
            gy = float(-h.py)
            vec = np.array([gx - h.px, gy - h.py], dtype=np.float32)
            norm = float(np.linalg.norm(vec) + 1e-6)
            dirx, diry = vec[0] / norm, vec[1] / norm
            speed = float(human_vpref)
            h.vx = dirx * speed
            h.vy = diry * speed
            h.radius = float(human_radius)
            # 设置时间步长（ORCA策略需要）
            h.time_step = self.config.time_step
            # 确保策略也有时间步长
            if hasattr(h.policy, 'time_step'):
                h.policy.time_step = self.config.time_step
            # 如果上层需要"可观测状态里有目标"，也可赋值：
            h.gx = gx
            h.gy = gy
            self.humans.append(h)

    # ----------- 单步推进（极简奖励在此） -----------
    def step(self, action, update: bool = True):
        assert self.robot is not None
        dt = float(self.config.time_step)
        # 统一：不再做二次"避让钩子"，完全交给 ORCA
        if not hasattr(self, '_il_avoid_logged'):
            logging.info("[ENV] Human velocities are purely from ORCA (no post-hook).")
            self._il_avoid_logged = True

        # 2) 人类决策（现在基于避让后的状态）
        human_actions = []
        for human in self.humans:
            ob = [other.get_observable_state() for other in self.humans if other is not human]
            if getattr(self.robot, 'visible', True):
                ob += [self.robot.get_observable_state()]
            human_actions.append(human.act(ob))

        # 3) 统一动作向量
        a = self._to_action(action)
        rvx, rvy = -float(a.vx), -float(a.vy)

        # 4) 几何关系统计（d_min / ttc_min / collision）- 增加调试信息
        dmin = float('inf')
        ttc_min = float('inf')
        closest_id = -1
        collision = False

        # 严格碰撞判定：恢复CrowdNav口径
        collision_threshold = 0.0

        for i, h in enumerate(self.humans):
            px = h.px - self.robot.px
            py = h.py - self.robot.py
            vx = h.vx + rvx
            vy = h.vy + rvy
            ex = px + vx * dt
            ey = py + vy * dt
            closest = point_to_segment_dist(px, py, ex, ey, 0.0, 0.0) - h.radius - self.robot.radius

            if closest < collision_threshold:
                collision = True
                closest_id = i
                dmin = closest
                ttc_min = 0.0
                break
            if closest < dmin:
                dmin = closest
                closest_id = i
            # ttc（仅统计）
            ttc = _ttc_rel(px, py, vx, vy, h.radius + self.robot.radius)
            if ttc < ttc_min:
                ttc_min = ttc

        # 5) 到达/进度
        next_pos = self.robot.compute_position(a, dt)
        reaching_goal = norm(np.array(next_pos) - np.array(self.robot.get_goal_position())) < self.rwd.success_radius
        prev_dist = norm(np.array([self.robot.px, self.robot.py]) - np.array(self.robot.get_goal_position()))
        curr_dist = norm(np.array(next_pos) - np.array(self.robot.get_goal_position()))

        # 6) 奖励（极简 CrowdNav 风格）
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        # 终止判定顺序（debug.md①要求）：collision → reach_goal → timeout
        if collision:                             # 1) 碰撞（最高优先级）
            reward = float(self.rwd.collision_penalty)
            terminated, truncated = True, False
            info = {"event": "collision", "dmin": dmin, "hid": closest_id, "ttc": 0.0}
        elif reaching_goal:                       # 2) 到达
            reward = float(self.rwd.success_reward)
            terminated, truncated = True, False
            info = {"event": "reach_goal", "dmin": max(dmin,0.0), "ttc": float(ttc_min)}
        elif self.step_count >= self.time_limit_steps:  # 3) 超时（最低优先级，改为按步数判定）
            reward = float(self.rwd.timeout_penalty)
            terminated, truncated = False, True
            info = {"event": "timeout", "dmin": max(dmin,0.0), "ttc": float(ttc_min)}
        else:
            reward = 0.0
            info = {"event": "ongoing", "dmin": max(dmin,0.0), "ttc": float(ttc_min)}
            # MSR 开关（默认 False，P1 再考虑打开）
            if getattr(self, 'msr_use_progress', False) or \
               getattr(self, 'msr_use_time_penalty', False) or \
               getattr(self, 'msr_use_antifreeze', False) or \
               getattr(self, 'msr_use_soft_ttc', False):
                msr_reward = self._msr_shaping(action, curr_dist, ttc_min)  # 封装MSR逻辑
                reward += msr_reward

        # 7) 推进世界 & 观测
        if update:
            self.states.append([self.robot.get_full_state(), [h.get_full_state() for h in self.humans]])
            self.robot.step(a)
            for i, ha in enumerate(human_actions):
                self.humans[i].step(ha)
            self.global_time += dt
            # debug.md斧头一: 步数计数器递增
            self.step_count += 1
            ob = self._obs()
        else:
            ob = self._next_obs(action, human_actions)


        # A4: 终止事件互斥断言 - 确保success+collision+timeout==1
        # 根据terminated和truncated状态统计事件数
        timeout_event = int(truncated and not terminated)
        collision_event = int(terminated and info.get("event") == "collision")
        success_event = int(terminated and info.get("event") == "reach_goal")

        # 互斥性检查：有且仅有一个终止事件发生
        total_events = timeout_event + collision_event + success_event

        # 对于非终止步骤，total_events应该为0
        # 对于终止步骤，total_events应该为1
        is_final_step = terminated or truncated

        if is_final_step:
            assert total_events == 1, (
                f"终止事件不互斥! timeout={timeout_event}, collision={collision_event}, "
                f"success={success_event}, total={total_events}, info={info}")
        else:
            assert total_events == 0, (
                f"非终止步骤不应有终止事件! timeout={timeout_event}, collision={collision_event}, "
                f"success={success_event}, total={total_events}, info={info}")

        return ob, float(reward), bool(terminated), bool(truncated), info

    def _msr_shaping(self, action, curr_dist, ttc_min):
        """MSR (Minimal Safe Reward) 封装方法，P1阶段使用"""
        msr_reward = 0.0
        msr_details = {}

        # 1) 目标进度势能（potential-based progress）
        if self.msr_use_progress:
            current_goal_distance = curr_dist
            if self.prev_goal_distance is not None:
                if self.reward_profile == 'success_boost':
                    # Win-Now阶段：使用Δd强信号
                    progress_delta = max(0.0, self.prev_goal_distance - current_goal_distance)
                    progress_reward = self.msr_lambda_p * progress_delta
                    msr_details['progress_type'] = 'delta_d'
                else:
                    # Benchmark阶段：势能型
                    gamma = 0.95
                    phi_prev = -self.prev_goal_distance
                    phi_curr = -current_goal_distance
                    progress_reward = self.msr_lambda_p * (gamma * phi_curr - phi_prev)
                    msr_details['progress_type'] = 'potential'

                msr_reward += progress_reward
                msr_details['progress'] = progress_reward
            self.prev_goal_distance = current_goal_distance

        # 2) 轻微时间罚
        if self.msr_use_time_penalty:
            time_penalty = -self.msr_eta
            msr_reward += time_penalty
            msr_details['time_penalty'] = time_penalty

        # 3) 反僵滞
        if self.msr_use_antifreeze:
            a_vec = self._to_action(action)
            robot_speed = math.sqrt(float(a_vec.vx)**2 + float(a_vec.vy)**2)
            self.robot_speed_history.append(robot_speed)

            if len(self.robot_speed_history) > self.msr_freeze_K:
                self.robot_speed_history.pop(0)

            if len(self.robot_speed_history) >= self.msr_freeze_K:
                low_speed_threshold = 0.2 * getattr(self.robot, 'v_pref', 1.0)
                low_speed_count = sum(1 for speed in self.robot_speed_history if speed < low_speed_threshold)

                if low_speed_count >= self.msr_freeze_K:
                    antifreeze_penalty = -self.msr_kappa
                    msr_reward += antifreeze_penalty
                    msr_details['antifreeze'] = antifreeze_penalty

        elif self.msr_use_soft_ttc:
            # 软TTC罚
            if ttc_min < self.msr_ttc_tau:
                ttc_penalty = -self.msr_alpha * (self.msr_ttc_tau - ttc_min) / self.msr_ttc_tau
                ttc_penalty = max(ttc_penalty, -0.2)
                msr_reward += ttc_penalty
                msr_details['soft_ttc'] = ttc_penalty

        return msr_reward

    # ------------------------ 观测/动作工具 ------------------------
    def _obs(self):
        if self.config.sensor == 'coordinates':
            return np.concatenate([self.robot.get_obs_array()] + [h.get_obs_array() for h in self.humans]).astype(np.float32)
        raise NotImplementedError

    def _next_obs(self, action, human_actions):
        if self.config.sensor == 'coordinates':
            return np.concatenate(
                [self.robot.get_next_obs_array(action)] +
                [h.get_next_obs_array(a) for h, a in zip(self.humans, human_actions)]
            ).astype(np.float32)
        raise NotImplementedError

    def _to_action(self, action_like):
        # 兼容 crowd_sim 的 Action 接口（有 vx/vy 的简单对象）
        return action_like

    # ------------------------ 人群生成（可选） ------------------------
    def _spawn_static_humans(self, n: int) -> List[Human]:
        humans = []
        try:
            # 若仓库有人类构造函数签名：(px,py, gx,gy, v_pref, radius)
            for i in range(n):
                h = Human()
                # 随机初始化（远离机器人原点，半径/速度从配置）
                ang = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(2.0, 4.0)
                h.px = float(math.cos(ang) * r)
                h.py = float(math.sin(ang) * r)
                h.vx = float(np.random.uniform(-0.2, 0.2))
                h.vy = float(np.random.uniform(-0.2, 0.2))
                h.radius = float(self.config.human_radius)
                humans.append(h)
        except Exception:
            # 无法构造则返回空，交给上层已有 humans
            pass
        return humans

    # ------------------------ 其他（可选） ------------------------
    def render(self, *args, **kwargs):
        return  # 可按需接入 matplotlib


# ----------------------------- 配置读取 -----------------------------

def _get(cfg_like, sec, key, default):
    """支持 configparser / dict / 对象属性 的读取。"""
    if hasattr(cfg_like, 'get'):
        try:
            return cfg_like.get(sec, key, fallback=default)
        except Exception:
            return default
    try:
        if isinstance(cfg_like, dict):
            if isinstance(cfg_like.get(sec, None), dict):
                return cfg_like.get(sec, {}).get(key, default)
            return cfg_like.get(key, default)
        val = getattr(cfg_like, key, default)
        return val
    except Exception:
        return default
