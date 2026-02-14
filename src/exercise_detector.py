from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class ExerciseDetector:
    exercise_classify_frames: int = 45
    exercise_min_confidence: float = 0.65
    trx_pike_gate_frames: int = 6

    exercise_type: Optional[str] = None
    exercise_locked: bool = False
    exercise_candidate: Optional[str] = None
    exercise_votes: dict[str, int] = field(default_factory=lambda: {"rdl": 0, "trx_pike": 0})
    classify_frame_count: int = 0
    trx_pike_gate_count: int = 0

    def set_exercise_type(self, exercise: str) -> Optional[str]:
        if exercise not in {"rdl", "trx_pike"}:
            return None
        self.exercise_type = exercise
        self.exercise_candidate = exercise
        self.exercise_locked = True
        return exercise

    @staticmethod
    def _exercise_scores(
        hip_angle: float,
        knee_angle: float,
        elbow_angle: float,
        body_angle: float,
        strap_confidence: float,
        hip_lift: float,
        ankle_above_hip: bool,
        ankle_below_hip: bool,
        wrist_below_shoulder: bool,
        hip_over_shoulder: bool,
        wrist_on_ground: bool,
    ) -> Tuple[float, float]:
        trx_score = 0.0
        if not np.isnan(hip_lift) and hip_lift > 0.22:
            trx_score += 1.6
        if not np.isnan(body_angle) and body_angle < 120.0:
            trx_score += 1.2
        if ankle_above_hip:
            trx_score += 0.8
        if not np.isnan(knee_angle) and knee_angle < 150.0:
            trx_score += 0.5
        if wrist_below_shoulder:
            trx_score += 0.4
        if hip_over_shoulder:
            trx_score += 0.8
        if wrist_on_ground:
            trx_score += 0.6
        if not np.isnan(elbow_angle) and elbow_angle > 150.0:
            trx_score += 0.2
        if strap_confidence >= 0.5:
            trx_score += 0.6

        rdl_score = 0.0
        if not np.isnan(hip_angle) and hip_angle < 150.0:
            rdl_score += 1.2
        if not np.isnan(knee_angle) and knee_angle > 150.0:
            rdl_score += 0.9
        if not np.isnan(body_angle) and body_angle > 120.0:
            rdl_score += 0.6
        if not np.isnan(elbow_angle) and elbow_angle > 150.0:
            rdl_score += 0.4
        if ankle_below_hip:
            rdl_score += 0.5
        if not np.isnan(hip_lift) and hip_lift < 0.2:
            rdl_score += 0.4
        if hip_over_shoulder:
            rdl_score -= 0.8
        if wrist_on_ground:
            rdl_score -= 0.4
        rdl_score = max(rdl_score, 0.0)

        trx_conf = min(trx_score / 5.4, 1.0)
        rdl_conf = min(rdl_score / 4.0, 1.0)
        return rdl_conf, trx_conf

    def confidence(
        self,
        hip_angle: float,
        knee_angle: float,
        elbow_angle: float,
        body_angle: float,
        strap_confidence: float,
        hip_lift: float,
        ankle_above_hip: bool,
        ankle_below_hip: bool,
        wrist_below_shoulder: bool,
        hip_over_shoulder: bool,
        wrist_on_ground: bool,
    ) -> float:
        rdl_conf, trx_conf = self._exercise_scores(
            hip_angle,
            knee_angle,
            elbow_angle,
            body_angle,
            strap_confidence,
            hip_lift,
            ankle_above_hip,
            ankle_below_hip,
            wrist_below_shoulder,
            hip_over_shoulder,
            wrist_on_ground,
        )
        return float(max(rdl_conf, trx_conf))

    def update(
        self,
        hip_angle: float,
        knee_angle: float,
        elbow_angle: float,
        body_angle: float,
        strap_confidence: float,
        hip_lift: float,
        ankle_above_hip: bool,
        ankle_below_hip: bool,
        wrist_below_shoulder: bool,
        hip_over_shoulder: bool,
        wrist_on_ground: bool,
    ) -> Optional[str]:
        if self.exercise_locked:
            return None

        if hip_over_shoulder and wrist_on_ground and not np.isnan(hip_lift) and hip_lift > 0.18:
            self.trx_pike_gate_count += 1
        else:
            self.trx_pike_gate_count = 0

        if self.trx_pike_gate_count >= self.trx_pike_gate_frames:
            return self.set_exercise_type("trx_pike")

        rdl_conf, trx_conf = self._exercise_scores(
            hip_angle,
            knee_angle,
            elbow_angle,
            body_angle,
            strap_confidence,
            hip_lift,
            ankle_above_hip,
            ankle_below_hip,
            wrist_below_shoulder,
            hip_over_shoulder,
            wrist_on_ground,
        )
        max_conf = max(rdl_conf, trx_conf)
        if max_conf < 0.3:
            self.exercise_candidate = None
            return None

        prediction = "trx_pike" if trx_conf > rdl_conf else "rdl"
        self.exercise_candidate = prediction
        self.classify_frame_count += 1
        self.exercise_votes[prediction] += 1

        vote_ratio = self.exercise_votes[prediction] / self.classify_frame_count
        if max_conf >= self.exercise_min_confidence and vote_ratio >= self.exercise_min_confidence:
            return self.set_exercise_type(prediction)

        if self.classify_frame_count >= self.exercise_classify_frames:
            final_choice = max(self.exercise_votes, key=self.exercise_votes.get)
            return self.set_exercise_type(final_choice)

        return None
