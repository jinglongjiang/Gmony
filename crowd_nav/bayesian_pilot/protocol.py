"""Reproducible nonstationary pedestrian behavior for the Bayesian pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


@dataclass(frozen=True)
class BehaviorProfile:
    name: str
    event_rate: float
    duration_steps: tuple[int, int]
    turn_degrees: tuple[float, float]
    slow_scale: tuple[float, float]
    event_weights: tuple[float, float, float, float]


PROFILES: Dict[str, BehaviorProfile] = {
    "nominal": BehaviorProfile(
        name="nominal",
        event_rate=0.0,
        duration_steps=(0, 0),
        turn_degrees=(0.0, 0.0),
        slow_scale=(1.0, 1.0),
        event_weights=(0.25, 0.25, 0.25, 0.25),
    ),
    "train_nonstationary": BehaviorProfile(
        name="train_nonstationary",
        event_rate=0.025,
        duration_steps=(2, 6),
        turn_degrees=(20.0, 50.0),
        slow_scale=(0.35, 0.70),
        event_weights=(0.30, 0.25, 0.225, 0.225),
    ),
    "heldout_nonstationary": BehaviorProfile(
        name="heldout_nonstationary",
        event_rate=0.035,
        duration_steps=(4, 9),
        turn_degrees=(55.0, 85.0),
        slow_scale=(0.15, 0.45),
        event_weights=(0.35, 0.20, 0.225, 0.225),
    ),
}

MODE_NAMES = ("nominal", "stop", "slow", "turn_left", "turn_right")
MODE_TO_ID = {name: index for index, name in enumerate(MODE_NAMES)}
EVENT_MODES = MODE_NAMES[1:]


class InterventionORCA(Policy):
    """Apply a scheduled behavior intervention to a normal ORCA action."""

    def __init__(self, config):
        super().__init__()
        self.base = ORCA()
        self.base.configure(config)
        self.trainable = False
        self.multiagent_training = True
        self.kinematics = "holonomic"
        self.mode = "nominal"
        self.scale = 1.0
        self.turn_radians = 0.0

    def configure(self, config):
        self.base.configure(config)

    @property
    def time_step(self):
        return getattr(self.base, "time_step", 0.25)

    @time_step.setter
    def time_step(self, value):
        if hasattr(self, "base"):
            self.base.time_step = value

    def set_intervention(
        self,
        mode: str,
        *,
        scale: float = 1.0,
        turn_radians: float = 0.0,
    ):
        if mode not in MODE_TO_ID:
            raise ValueError(f"Unknown intervention mode: {mode}")
        self.mode = mode
        self.scale = float(scale)
        self.turn_radians = float(turn_radians)

    def reset(self):
        self.mode = "nominal"
        self.scale = 1.0
        self.turn_radians = 0.0
        self.base.sim = None
        self.base._last_pref_vel = None

    def predict(self, state):
        action = self.base.predict(state)
        vx, vy = float(action.vx), float(action.vy)
        if self.mode == "stop":
            return ActionXY(0.0, 0.0)
        if self.mode == "slow":
            return ActionXY(vx * self.scale, vy * self.scale)
        if self.mode in {"turn_left", "turn_right"}:
            angle = self.turn_radians
            if self.mode == "turn_right":
                angle = -angle
            c, s = float(np.cos(angle)), float(np.sin(angle))
            return ActionXY(c * vx - s * vy, s * vx + c * vy)
        return action


@dataclass
class _ActiveEvent:
    mode: str = "nominal"
    remaining: int = 0
    scale: float = 1.0
    turn_radians: float = 0.0


class BehaviorScheduler:
    """Independent per-pedestrian Markov event scheduler."""

    def __init__(self, profile: BehaviorProfile, seed: int):
        self.profile = profile
        self.rng = np.random.default_rng(int(seed))
        self.events: List[_ActiveEvent] = []
        self.counts = {name: 0 for name in MODE_NAMES}

    def reset(self, n_pedestrians: int):
        self.events = [_ActiveEvent() for _ in range(int(n_pedestrians))]
        self.counts = {name: 0 for name in MODE_NAMES}

    def _sample_event(self) -> _ActiveEvent:
        mode = str(
            self.rng.choice(
                EVENT_MODES,
                p=np.asarray(self.profile.event_weights, dtype=np.float64),
            )
        )
        lo, hi = self.profile.duration_steps
        duration = int(self.rng.integers(lo, hi + 1))
        scale = float(self.rng.uniform(*self.profile.slow_scale))
        turn = float(np.deg2rad(self.rng.uniform(*self.profile.turn_degrees)))
        return _ActiveEvent(
            mode=mode,
            remaining=duration,
            scale=scale,
            turn_radians=turn,
        )

    def advance(self, policies: Iterable[InterventionORCA]) -> List[int]:
        policies = list(policies)
        if len(policies) != len(self.events):
            raise ValueError("Scheduler and policy counts do not match")

        mode_ids = []
        for index, policy in enumerate(policies):
            event = self.events[index]
            if event.remaining > 0:
                event.remaining -= 1
            else:
                event = _ActiveEvent()
                if (
                    self.profile.event_rate > 0.0
                    and self.rng.random() < self.profile.event_rate
                ):
                    event = self._sample_event()
                    self.counts[event.mode] += 1
                self.events[index] = event

            policy.set_intervention(
                event.mode,
                scale=event.scale,
                turn_radians=event.turn_radians,
            )
            mode_ids.append(MODE_TO_ID[event.mode])
        return mode_ids

