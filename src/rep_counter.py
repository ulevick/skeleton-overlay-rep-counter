from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .pose_analyzer import AnalyzerConfig


@dataclass
class RepCounter:
    stage: str = "up"
    rep_count: int = 0
    rep_events: List[float] = field(default_factory=list)
    last_rep_time: float = -1.0
    knee_warn_counter: int = 0
    back_warn_counter: int = 0
    elbow_warn_counter: int = 0
    pike_up_counter: int = 0
    pike_down_counter: int = 0

    def reset_for_exercise(self, exercise: str) -> None:
        self.rep_count = 0
        self.rep_events = []
        self.last_rep_time = -1.0
        self.knee_warn_counter = 0
        self.back_warn_counter = 0
        self.elbow_warn_counter = 0
        self.pike_up_counter = 0
        self.pike_down_counter = 0
        self.stage = "up" if exercise == "rdl" else "plank"

    def process_rdl(
        self,
        *,
        hip_angle_smooth: float,
        knee_angle_smooth: float,
        back_angle_smooth: float,
        tracking_ok: bool,
        timestamp_s: float,
        config: "AnalyzerConfig",
    ) -> List[str]:
        warnings: List[str] = []
        should_check = (
            tracking_ok
            and not np.isnan(hip_angle_smooth)
            and hip_angle_smooth < config.warning_hip_angle
        )

        if (
            should_check
            and not np.isnan(knee_angle_smooth)
            and knee_angle_smooth < config.knee_warning_angle
        ):
            self.knee_warn_counter += 1
        else:
            self.knee_warn_counter = 0

        if (
            should_check
            and not np.isnan(back_angle_smooth)
            and back_angle_smooth < config.back_warning_angle
        ):
            self.back_warn_counter += 1
        else:
            self.back_warn_counter = 0

        if self.knee_warn_counter >= config.warning_hold_frames:
            warnings.append("Knees too bent")
        if self.back_warn_counter >= config.warning_hold_frames:
            warnings.append("Back rounded")

        if tracking_ok and not np.isnan(hip_angle_smooth):
            if hip_angle_smooth < config.hip_down_angle and self.stage != "down":
                self.stage = "down"
            elif hip_angle_smooth > config.hip_up_angle and self.stage == "down":
                if self.last_rep_time < 0 or (timestamp_s - self.last_rep_time) >= config.min_rep_interval_s:
                    self.stage = "up"
                    self.rep_count += 1
                    self.rep_events.append(timestamp_s)
                    self.last_rep_time = timestamp_s

        return warnings

    def process_trx_pike(
        self,
        *,
        elbow_angle_smooth: float,
        body_angle_smooth: float,
        knee_angle_smooth: float,
        hip_lift_smooth: float,
        tracking_ok: bool,
        timestamp_s: float,
        config: "AnalyzerConfig",
    ) -> List[str]:
        warnings: List[str] = []

        hip_valid = not np.isnan(hip_lift_smooth)
        body_valid = not np.isnan(body_angle_smooth)
        hip_up = hip_valid and hip_lift_smooth > config.trx_pike_up_lift
        hip_down = hip_valid and hip_lift_smooth < config.trx_pike_down_lift
        body_up = body_valid and body_angle_smooth < config.trx_pike_up_angle
        body_down = body_valid and body_angle_smooth > config.trx_pike_down_angle

        if hip_valid and body_valid:
            is_pike_up = hip_up and body_up
            is_pike_down = hip_down and body_down
        else:
            is_pike_up = hip_up or body_up
            is_pike_down = hip_down or body_down

        if tracking_ok and is_pike_up:
            self.pike_up_counter += 1
        else:
            self.pike_up_counter = 0

        if tracking_ok and is_pike_down:
            self.pike_down_counter += 1
        else:
            self.pike_down_counter = 0

        if self.stage != "pike" and self.pike_up_counter >= config.trx_pike_stage_frames:
            self.stage = "pike"
            self.pike_down_counter = 0
        elif self.stage == "pike" and self.pike_down_counter >= config.trx_pike_stage_frames:
            if self.last_rep_time < 0 or (timestamp_s - self.last_rep_time) >= config.min_rep_interval_s:
                self.stage = "plank"
                self.rep_count += 1
                self.rep_events.append(timestamp_s)
                self.last_rep_time = timestamp_s
            self.pike_up_counter = 0

        pike_signal = False
        if not np.isnan(body_angle_smooth) and not np.isnan(hip_lift_smooth):
            pike_signal = (
                body_angle_smooth < config.trx_pike_warning_angle
                and hip_lift_smooth > config.trx_pike_down_lift
            )
        elif not np.isnan(body_angle_smooth):
            pike_signal = body_angle_smooth < config.trx_pike_warning_angle
        elif not np.isnan(hip_lift_smooth):
            pike_signal = hip_lift_smooth > config.trx_pike_down_lift

        should_check = tracking_ok and pike_signal and (
            self.stage == "pike" or self.pike_up_counter >= config.trx_pike_stage_frames
        )

        if (
            should_check
            and not np.isnan(elbow_angle_smooth)
            and elbow_angle_smooth < config.trx_pike_elbow_warning_angle
        ):
            self.elbow_warn_counter += 1
        else:
            self.elbow_warn_counter = 0

        knee_threshold = config.trx_pike_knee_warning_angle - 5.0
        knee_ok_signal = not np.isnan(hip_lift_smooth) and hip_lift_smooth > config.trx_pike_up_lift * 0.9
        if (
            should_check
            and not np.isnan(knee_angle_smooth)
            and knee_ok_signal
            and knee_angle_smooth < knee_threshold
        ):
            self.knee_warn_counter += 1
        else:
            self.knee_warn_counter = 0

        if self.elbow_warn_counter >= config.warning_hold_frames:
            warnings.append("Elbows bent")
        if self.knee_warn_counter >= config.warning_hold_frames:
            warnings.append("Knees slightly bent")

        return warnings
