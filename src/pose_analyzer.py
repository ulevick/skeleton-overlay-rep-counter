from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .utils import calculate_angle, landmark_xy, landmark_xy_visibility, safe_mean, visibility_score

_HAS_SOLUTIONS = hasattr(mp, "solutions")
if not _HAS_SOLUTIONS:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

LANDMARK_INDEX = {
    "NOSE": 0,
    "LEFT_EAR": 7,
    "RIGHT_EAR": 8,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
}

LEFT_SIDE_INDICES = {
    LANDMARK_INDEX["LEFT_SHOULDER"],
    LANDMARK_INDEX["LEFT_ELBOW"],
    LANDMARK_INDEX["LEFT_WRIST"],
    LANDMARK_INDEX["LEFT_HIP"],
    LANDMARK_INDEX["LEFT_KNEE"],
    LANDMARK_INDEX["LEFT_ANKLE"],
    LANDMARK_INDEX["LEFT_EAR"],
}
RIGHT_SIDE_INDICES = {
    LANDMARK_INDEX["RIGHT_SHOULDER"],
    LANDMARK_INDEX["RIGHT_ELBOW"],
    LANDMARK_INDEX["RIGHT_WRIST"],
    LANDMARK_INDEX["RIGHT_HIP"],
    LANDMARK_INDEX["RIGHT_KNEE"],
    LANDMARK_INDEX["RIGHT_ANKLE"],
    LANDMARK_INDEX["RIGHT_EAR"],
}
CENTER_INDICES = {LANDMARK_INDEX["NOSE"]}

POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]


@dataclass
class AnalyzerConfig:
    hip_down_angle: float = 110.0
    hip_up_angle: float = 160.0
    knee_warning_angle: float = 150.0
    back_warning_angle: float = 160.0
    warning_hip_angle: float = 150.0
    warning_hold_frames: int = 5
    visibility_threshold: float = 0.5
    smoothing_window: int = 7
    model_path: Optional[str] = None
    model_complexity: int = 2
    min_rep_interval_s: float = 0.6
    exercise_mode: str = "auto"  # auto, rdl, trx_pike
    exercise_classify_frames: int = 45
    exercise_min_confidence: float = 0.65
    strap_detection: bool = True
    strap_detection_scale: float = 0.5
    trx_pike_up_angle: float = 110.0
    trx_pike_down_angle: float = 160.0
    trx_pike_up_lift: float = 0.35
    trx_pike_down_lift: float = 0.1
    trx_pike_warning_angle: float = 150.0
    trx_pike_elbow_warning_angle: float = 160.0
    trx_pike_knee_warning_angle: float = 160.0
    trx_pike_gate_frames: int = 6
    trx_pike_stage_frames: int = 5
    suppress_far_side: bool = True
    side_view_visibility_gap: float = 0.2


class PoseAnalyzer:
    def __init__(self, config: Optional[AnalyzerConfig] = None) -> None:
        self.config = config or AnalyzerConfig()
        self.use_tasks = not _HAS_SOLUTIONS
        if self.use_tasks:
            if not self.config.model_path:
                raise RuntimeError("MediaPipe solutions API not available. Provide model_path for tasks API.")
            base_options = mp_tasks.BaseOptions(model_asset_path=self.config.model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.pose = vision.PoseLandmarker.create_from_options(options)
            self.drawer = None
            self.drawing_styles = None
        else:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=self.config.model_complexity,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.drawer = mp.solutions.drawing_utils
            self.drawing_styles = mp.solutions.drawing_styles

        self.rep_count = 0
        self.stage = "up"
        self.hip_angle_history: deque[float] = deque(maxlen=self.config.smoothing_window)
        self.knee_angle_history: deque[float] = deque(maxlen=self.config.smoothing_window)
        self.back_angle_history: deque[float] = deque(maxlen=self.config.smoothing_window)
        self.rep_events: List[float] = []
        self.last_rep_time = -1.0
        self.knee_warn_counter = 0
        self.back_warn_counter = 0
        self.elbow_warn_counter = 0
        self.exercise_type: Optional[str] = None
        self.exercise_locked = False
        self.exercise_votes = {"rdl": 0, "trx_pike": 0}
        self.classify_frame_count = 0
        self.exercise_candidate: Optional[str] = None
        self.trx_pike_gate_count = 0
        self.pike_up_counter = 0
        self.pike_down_counter = 0
        self.elbow_angle_history: deque[float] = deque(maxlen=self.config.smoothing_window)
        self.body_angle_history: deque[float] = deque(maxlen=self.config.smoothing_window)
        self.hip_lift_history: deque[float] = deque(maxlen=self.config.smoothing_window)

        if self.config.exercise_mode != "auto":
            self._set_exercise_type(self.config.exercise_mode)

    def close(self) -> None:
        self.pose.close()

    def _detect_landmarks(self, image_rgb: np.ndarray, timestamp_s: float):
        if self.use_tasks:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self.pose.detect_for_video(mp_image, int(timestamp_s * 1000))
            landmarks = results.pose_landmarks[0] if results.pose_landmarks else None
            return landmarks, landmarks

        results = self.pose.process(image_rgb)
        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
        return landmarks, results.pose_landmarks

    def _base_record(self, timestamp_s: float) -> Dict[str, object]:
        return {
            "timestamp_s": timestamp_s,
            "side": None,
            "hip_angle": None,
            "hip_angle_smooth": None,
            "knee_angle": None,
            "back_angle": None,
            "knee_angle_smooth": None,
            "back_angle_smooth": None,
            "elbow_angle": None,
            "elbow_angle_smooth": None,
            "body_angle": None,
            "body_angle_smooth": None,
            "hip_lift": None,
            "hip_lift_smooth": None,
            "hip_angle_display": "n/a",
            "knee_angle_display": "n/a",
            "back_angle_display": "n/a",
            "elbow_angle_display": "n/a",
            "body_angle_display": "n/a",
            "tracking_score": None,
            "tracking_ok": False,
            "tracking_display": "low",
            "exercise": "detecting",
            "exercise_confidence": 0.0,
            "exercise_locked": False,
            "strap_confidence": 0.0,
            "stage": self.stage,
            "rep_count": self.rep_count,
            "warnings": [],
            "pose_detected": False,
        }

    def _draw_pose(self, frame: np.ndarray, draw_payload, draw_side: Optional[str] = None) -> None:
        if not draw_payload:
            return
        if self.use_tasks:
            landmarks = draw_payload
        else:
            landmarks = draw_payload.landmark
        allowed_indices = None
        if draw_side:
            allowed_indices = self._allowed_indices_for_side(draw_side)
        self._draw_landmarks_custom(frame, landmarks, allowed_indices)

    def process_frame(
        self, frame: np.ndarray, timestamp_s: float, with_overlay: bool = True
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks, draw_payload = self._detect_landmarks(image_rgb, timestamp_s)
        record = self._base_record(timestamp_s)

        if not landmarks:
            annotated, _ = self._draw_panel(
                frame,
                ["Pose not detected"],
                (15, 15),
                color=(40, 40, 40),
                alpha=0.55,
            )
            return annotated, record

        record["pose_detected"] = True
        side, left_score, right_score = self._select_side(landmarks)
        record["side"] = side

        if side == "left":
            shoulder = LANDMARK_INDEX["LEFT_SHOULDER"]
            elbow = LANDMARK_INDEX["LEFT_ELBOW"]
            wrist = LANDMARK_INDEX["LEFT_WRIST"]
            hip = LANDMARK_INDEX["LEFT_HIP"]
            knee = LANDMARK_INDEX["LEFT_KNEE"]
            ankle = LANDMARK_INDEX["LEFT_ANKLE"]
            ear = LANDMARK_INDEX["LEFT_EAR"]
        else:
            shoulder = LANDMARK_INDEX["RIGHT_SHOULDER"]
            elbow = LANDMARK_INDEX["RIGHT_ELBOW"]
            wrist = LANDMARK_INDEX["RIGHT_WRIST"]
            hip = LANDMARK_INDEX["RIGHT_HIP"]
            knee = LANDMARK_INDEX["RIGHT_KNEE"]
            ankle = LANDMARK_INDEX["RIGHT_ANKLE"]
            ear = LANDMARK_INDEX["RIGHT_EAR"]

        nose = LANDMARK_INDEX["NOSE"]

        required_indices = [shoulder, elbow, wrist, hip, knee, ankle, ear, nose]
        tracking_score = visibility_score(landmarks, required_indices)
        tracking_ok = tracking_score >= self.config.visibility_threshold

        shoulder_xy = landmark_xy(landmarks, shoulder)
        elbow_xy = landmark_xy(landmarks, elbow)
        wrist_xy = landmark_xy(landmarks, wrist)
        hip_xy = landmark_xy(landmarks, hip)
        knee_xy = landmark_xy(landmarks, knee)
        ankle_xy = landmark_xy(landmarks, ankle)
        ear_xy, ear_vis = landmark_xy_visibility(landmarks, ear)
        nose_xy, _ = landmark_xy_visibility(landmarks, nose)
        head_xy = ear_xy if ear_vis >= self.config.visibility_threshold else nose_xy

        hip_angle = calculate_angle(shoulder_xy, hip_xy, knee_xy)
        elbow_angle = calculate_angle(shoulder_xy, elbow_xy, wrist_xy)
        body_angle = calculate_angle(shoulder_xy, hip_xy, ankle_xy)
        knee_angle = calculate_angle(hip_xy, knee_xy, ankle_xy)
        back_angle = calculate_angle(hip_xy, shoulder_xy, head_xy)

        torso_length = float(np.linalg.norm(np.array(shoulder_xy) - np.array(hip_xy)))
        if torso_length > 1e-6:
            hip_lift = float((shoulder_xy[1] - hip_xy[1]) / torso_length)
        else:
            hip_lift = float("nan")

        if not np.isnan(hip_angle):
            self.hip_angle_history.append(hip_angle)
        if not np.isnan(elbow_angle):
            self.elbow_angle_history.append(elbow_angle)
        if not np.isnan(body_angle):
            self.body_angle_history.append(body_angle)
        if not np.isnan(hip_lift):
            self.hip_lift_history.append(hip_lift)
        if not np.isnan(knee_angle):
            self.knee_angle_history.append(knee_angle)
        if not np.isnan(back_angle):
            self.back_angle_history.append(back_angle)

        hip_angle_smooth = safe_mean(self.hip_angle_history)
        elbow_angle_smooth = safe_mean(self.elbow_angle_history)
        body_angle_smooth = safe_mean(self.body_angle_history)
        hip_lift_smooth = safe_mean(self.hip_lift_history)
        knee_angle_smooth = safe_mean(self.knee_angle_history)
        back_angle_smooth = safe_mean(self.back_angle_history)

        record["hip_angle"] = hip_angle
        record["hip_angle_smooth"] = hip_angle_smooth
        record["elbow_angle"] = elbow_angle
        record["elbow_angle_smooth"] = elbow_angle_smooth
        record["body_angle"] = body_angle
        record["body_angle_smooth"] = body_angle_smooth
        record["hip_lift"] = hip_lift
        record["hip_lift_smooth"] = hip_lift_smooth
        record["knee_angle"] = knee_angle
        record["back_angle"] = back_angle
        record["knee_angle_smooth"] = knee_angle_smooth
        record["back_angle_smooth"] = back_angle_smooth
        record["tracking_score"] = round(tracking_score, 3)
        record["tracking_ok"] = tracking_ok
        record["hip_angle_display"] = self._fmt_angle(hip_angle_smooth)
        record["knee_angle_display"] = self._fmt_angle(knee_angle_smooth)
        record["back_angle_display"] = self._fmt_angle(back_angle_smooth)
        record["elbow_angle_display"] = self._fmt_angle(elbow_angle_smooth)
        record["body_angle_display"] = self._fmt_angle(body_angle_smooth)
        record["tracking_display"] = "ok" if tracking_ok else "low"

        ankle_above_hip = ankle_xy[1] < hip_xy[1] - 0.05
        ankle_below_hip = ankle_xy[1] > hip_xy[1] + 0.05
        wrist_below_shoulder = wrist_xy[1] > shoulder_xy[1] + 0.05
        hip_over_shoulder = hip_xy[1] < shoulder_xy[1] - 0.03
        wrist_on_ground = wrist_xy[1] > 0.8

        strap_confidence = self._estimate_trx_straps(frame, landmarks)
        record["strap_confidence"] = strap_confidence
        self._update_exercise_type(
            hip_angle_smooth,
            knee_angle_smooth,
            elbow_angle_smooth,
            body_angle_smooth,
            strap_confidence,
            hip_lift_smooth,
            ankle_above_hip,
            ankle_below_hip,
            wrist_below_shoulder,
            hip_over_shoulder,
            wrist_on_ground,
        )
        record["exercise"] = self.exercise_type or self.exercise_candidate or "detecting"
        record["exercise_confidence"] = self._exercise_confidence(
            hip_angle_smooth,
            knee_angle_smooth,
            elbow_angle_smooth,
            body_angle_smooth,
            strap_confidence,
            hip_lift_smooth,
            ankle_above_hip,
            ankle_below_hip,
            wrist_below_shoulder,
            hip_over_shoulder,
            wrist_on_ground,
        )
        record["exercise_locked"] = self.exercise_locked

        warnings = []
        if self.exercise_type == "rdl":
            warnings = self._process_rdl(
                hip_angle_smooth,
                knee_angle_smooth,
                back_angle_smooth,
                tracking_ok,
                timestamp_s,
            )
        elif self.exercise_type == "trx_pike":
            warnings = self._process_trx_pike(
                elbow_angle_smooth,
                body_angle_smooth,
                knee_angle_smooth,
                hip_lift_smooth,
                tracking_ok,
                timestamp_s,
            )

        record["warnings"] = warnings
        record["stage"] = self.stage
        record["rep_count"] = self.rep_count

        annotated = frame.copy()
        draw_side = None
        if self.config.suppress_far_side:
            if abs(left_score - right_score) >= self.config.side_view_visibility_gap:
                draw_side = side
        self._draw_pose(annotated, draw_payload, draw_side)

        if with_overlay:
            exercise_title = (
                self.exercise_type.replace("_", " ").title()
                if self.exercise_type
                else "Detecting exercise"
            )
            overlay_lines = [
                exercise_title,
                f"Reps: {self.rep_count}",
                f"Hip angle: {self._fmt_angle(hip_angle_smooth)}",
                f"Elbow angle: {self._fmt_angle(elbow_angle_smooth)}",
                f"Body angle: {self._fmt_angle(body_angle_smooth)}",
                f"Stage: {self.stage}",
                f"Tracking: {'ok' if tracking_ok else 'low'}",
            ]
            annotated, panel_rect = self._draw_panel(
                annotated,
                overlay_lines,
                (15, 15),
                color=(20, 20, 20),
                alpha=0.5,
            )

            if warnings:
                warning_origin = (15, panel_rect[1] + panel_rect[3] + 12)
                annotated, _ = self._draw_panel(
                    annotated,
                    ["Warning:"] + warnings,
                    warning_origin,
                    color=(20, 20, 90),
                    alpha=0.55,
                )

        return annotated, record

    def _set_exercise_type(self, exercise: str) -> None:
        if exercise not in {"rdl", "trx_pike"}:
            return
        self.exercise_type = exercise
        self.exercise_candidate = exercise
        self.exercise_locked = True
        self.rep_count = 0
        self.rep_events = []
        self.last_rep_time = -1.0
        self.knee_warn_counter = 0
        self.back_warn_counter = 0
        self.elbow_warn_counter = 0
        self.stage = "up" if exercise == "rdl" else "plank"

    def _exercise_scores(
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

    def _exercise_confidence(
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

    def _update_exercise_type(
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
    ) -> None:
        if self.exercise_locked:
            return

        if hip_over_shoulder and wrist_on_ground and not np.isnan(hip_lift) and hip_lift > 0.18:
            self.trx_pike_gate_count += 1
        else:
            self.trx_pike_gate_count = 0

        if self.trx_pike_gate_count >= self.config.trx_pike_gate_frames:
            self._set_exercise_type("trx_pike")
            return

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
            return

        prediction = "trx_pike" if trx_conf > rdl_conf else "rdl"
        self.exercise_candidate = prediction
        self.classify_frame_count += 1
        self.exercise_votes[prediction] += 1

        vote_ratio = self.exercise_votes[prediction] / self.classify_frame_count
        if max_conf >= self.config.exercise_min_confidence and vote_ratio >= self.config.exercise_min_confidence:
            self._set_exercise_type(prediction)
            return

        if self.classify_frame_count >= self.config.exercise_classify_frames:
            final_choice = max(self.exercise_votes, key=self.exercise_votes.get)
            self._set_exercise_type(final_choice)

    def _process_rdl(
        self,
        hip_angle_smooth: float,
        knee_angle_smooth: float,
        back_angle_smooth: float,
        tracking_ok: bool,
        timestamp_s: float,
    ) -> List[str]:
        warnings: List[str] = []
        should_check = (
            tracking_ok
            and not np.isnan(hip_angle_smooth)
            and hip_angle_smooth < self.config.warning_hip_angle
        )

        if (
            should_check
            and not np.isnan(knee_angle_smooth)
            and knee_angle_smooth < self.config.knee_warning_angle
        ):
            self.knee_warn_counter += 1
        else:
            self.knee_warn_counter = 0

        if (
            should_check
            and not np.isnan(back_angle_smooth)
            and back_angle_smooth < self.config.back_warning_angle
        ):
            self.back_warn_counter += 1
        else:
            self.back_warn_counter = 0

        if self.knee_warn_counter >= self.config.warning_hold_frames:
            warnings.append("Knees too bent")
        if self.back_warn_counter >= self.config.warning_hold_frames:
            warnings.append("Back rounded")

        if tracking_ok and not np.isnan(hip_angle_smooth):
            if hip_angle_smooth < self.config.hip_down_angle and self.stage != "down":
                self.stage = "down"
            elif hip_angle_smooth > self.config.hip_up_angle and self.stage == "down":
                if self.last_rep_time < 0 or (timestamp_s - self.last_rep_time) >= self.config.min_rep_interval_s:
                    self.stage = "up"
                    self.rep_count += 1
                    self.rep_events.append(timestamp_s)
                    self.last_rep_time = timestamp_s

        return warnings

    def _process_trx_pike(
        self,
        elbow_angle_smooth: float,
        body_angle_smooth: float,
        knee_angle_smooth: float,
        hip_lift_smooth: float,
        tracking_ok: bool,
        timestamp_s: float,
    ) -> List[str]:
        warnings: List[str] = []

        hip_valid = not np.isnan(hip_lift_smooth)
        body_valid = not np.isnan(body_angle_smooth)
        hip_up = hip_valid and hip_lift_smooth > self.config.trx_pike_up_lift
        hip_down = hip_valid and hip_lift_smooth < self.config.trx_pike_down_lift
        body_up = body_valid and body_angle_smooth < self.config.trx_pike_up_angle
        body_down = body_valid and body_angle_smooth > self.config.trx_pike_down_angle

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

        if self.stage != "pike" and self.pike_up_counter >= self.config.trx_pike_stage_frames:
            self.stage = "pike"
            self.pike_down_counter = 0
        elif self.stage == "pike" and self.pike_down_counter >= self.config.trx_pike_stage_frames:
            if self.last_rep_time < 0 or (timestamp_s - self.last_rep_time) >= self.config.min_rep_interval_s:
                self.stage = "plank"
                self.rep_count += 1
                self.rep_events.append(timestamp_s)
                self.last_rep_time = timestamp_s
            self.pike_up_counter = 0

        should_check = tracking_ok and (
            (not np.isnan(body_angle_smooth) and body_angle_smooth < self.config.trx_pike_warning_angle)
            or (not np.isnan(hip_lift_smooth) and hip_lift_smooth > self.config.trx_pike_down_lift)
        )

        if (
            should_check
            and not np.isnan(elbow_angle_smooth)
            and elbow_angle_smooth < self.config.trx_pike_elbow_warning_angle
        ):
            self.elbow_warn_counter += 1
        else:
            self.elbow_warn_counter = 0

        if (
            should_check
            and not np.isnan(knee_angle_smooth)
            and knee_angle_smooth < self.config.trx_pike_knee_warning_angle
        ):
            self.knee_warn_counter += 1
        else:
            self.knee_warn_counter = 0

        if self.elbow_warn_counter >= self.config.warning_hold_frames:
            warnings.append("Elbows bent")
        if self.knee_warn_counter >= self.config.warning_hold_frames:
            warnings.append("Knees bent")

        return warnings

    def _estimate_trx_straps(self, frame: np.ndarray, landmarks) -> float:
        if not self.config.strap_detection:
            return 0.0
        wrist_conf = self._estimate_straps_for_pair(landmarks, "LEFT_WRIST", "RIGHT_WRIST", frame)
        ankle_conf = self._estimate_straps_for_pair(landmarks, "LEFT_ANKLE", "RIGHT_ANKLE", frame)
        return max(wrist_conf, ankle_conf)

    def _estimate_straps_for_pair(
        self, landmarks, left_key: str, right_key: str, frame: np.ndarray
    ) -> float:
        left_idx = LANDMARK_INDEX[left_key]
        right_idx = LANDMARK_INDEX[right_key]
        left_vis = self._visibility(landmarks[left_idx])
        right_vis = self._visibility(landmarks[right_idx])
        if left_vis < self.config.visibility_threshold or right_vis < self.config.visibility_threshold:
            return 0.0

        height, width = frame.shape[:2]
        scale = self.config.strap_detection_scale
        scaled = frame
        if scale != 1.0:
            scaled = cv2.resize(frame, (int(width * scale), int(height * scale)))
        sh, sw = scaled.shape[:2]

        left_pt = (
            int(landmarks[left_idx].x * width * scale),
            int(landmarks[left_idx].y * height * scale),
        )
        right_pt = (
            int(landmarks[right_idx].x * width * scale),
            int(landmarks[right_idx].y * height * scale),
        )

        y_max = max(left_pt[1], right_pt[1])
        if y_max < 20:
            return 0.0

        margin_x = int(0.2 * sw)
        x_min = max(min(left_pt[0], right_pt[0]) - margin_x, 0)
        x_max = min(max(left_pt[0], right_pt[0]) + margin_x, sw - 1)
        y_max = min(y_max + int(0.05 * sh), sh - 1)
        roi = scaled[0:y_max, x_min:x_max]
        if roi.size == 0 or roi.shape[0] < 30 or roi.shape[1] < 30:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=40,
            minLineLength=int(0.4 * roi.shape[0]),
            maxLineGap=12,
        )
        if lines is None:
            return 0.0

        left_roi = (left_pt[0] - x_min, left_pt[1])
        right_roi = (right_pt[0] - x_min, right_pt[1])
        left_hit = False
        right_hit = False
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if min(y1, y2) > left_roi[1] - 10:
                continue
            if self._point_distance((x1, y1), left_roi) < 25 or self._point_distance((x2, y2), left_roi) < 25:
                left_hit = True
            if self._point_distance((x1, y1), right_roi) < 25 or self._point_distance((x2, y2), right_roi) < 25:
                right_hit = True

        if left_hit and right_hit:
            return 1.0
        if left_hit or right_hit:
            return 0.5
        return 0.0

    @staticmethod
    def _point_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _select_side(self, landmarks) -> Tuple[str, float, float]:
        left_indices = [
            LANDMARK_INDEX["LEFT_SHOULDER"],
            LANDMARK_INDEX["LEFT_HIP"],
            LANDMARK_INDEX["LEFT_KNEE"],
            LANDMARK_INDEX["LEFT_ANKLE"],
            LANDMARK_INDEX["LEFT_EAR"],
        ]
        right_indices = [
            LANDMARK_INDEX["RIGHT_SHOULDER"],
            LANDMARK_INDEX["RIGHT_HIP"],
            LANDMARK_INDEX["RIGHT_KNEE"],
            LANDMARK_INDEX["RIGHT_ANKLE"],
            LANDMARK_INDEX["RIGHT_EAR"],
        ]

        left_score = visibility_score(landmarks, left_indices)
        right_score = visibility_score(landmarks, right_indices)

        if left_score >= right_score:
            return "left", left_score, right_score
        return "right", left_score, right_score

    def _allowed_indices_for_side(self, side: str) -> set[int]:
        if side == "left":
            return LEFT_SIDE_INDICES | CENTER_INDICES
        if side == "right":
            return RIGHT_SIDE_INDICES | CENTER_INDICES
        return set()

    def _draw_landmarks_custom(
        self, frame: np.ndarray, landmarks, allowed_indices: Optional[set[int]] = None
    ) -> None:
        height, width = frame.shape[:2]
        for start_idx, end_idx in POSE_CONNECTIONS:
            if allowed_indices and (start_idx not in allowed_indices or end_idx not in allowed_indices):
                continue
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            if self._visibility(start) < self.config.visibility_threshold and self._visibility(end) < self.config.visibility_threshold:
                continue
            x1, y1 = self._to_pixel(start, width, height)
            x2, y2 = self._to_pixel(end, width, height)
            cv2.line(frame, (x1, y1), (x2, y2), (80, 220, 80), 2)

        for idx, landmark in enumerate(landmarks):
            if allowed_indices and idx not in allowed_indices:
                continue
            if self._visibility(landmark) < self.config.visibility_threshold:
                continue
            x, y = self._to_pixel(landmark, width, height)
            cv2.circle(frame, (x, y), 3, (0, 200, 255), -1)

    @staticmethod
    def _visibility(landmark) -> float:
        visibility = getattr(landmark, "visibility", None)
        if visibility is None:
            visibility = getattr(landmark, "presence", 1.0)
        return float(visibility)

    @staticmethod
    def _to_pixel(landmark, width: int, height: int) -> Tuple[int, int]:
        return int(landmark.x * width), int(landmark.y * height)

    @staticmethod
    def _fmt_angle(value: Optional[float]) -> str:
        if value is None or np.isnan(value):
            return "n/a"
        return f"{value:.1f} deg"

    @staticmethod
    def _draw_panel(
        frame: np.ndarray,
        lines: List[str],
        origin: Tuple[int, int],
        color: Tuple[int, int, int],
        alpha: float = 0.6,
        font_scale: float = 0.6,
        thickness: int = 2,
        padding: Tuple[int, int] = (12, 10),
        line_gap: int = 6,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX
        max_width = 0
        line_heights: List[int] = []
        for line in lines:
            (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, text_width)
            line_heights.append(text_height + baseline)

        pad_x, pad_y = padding
        width = max_width + pad_x * 2
        height = sum(line_heights) + line_gap * (len(lines) - 1) + pad_y * 2

        frame_height, frame_width = frame.shape[:2]
        width = min(width, frame_width - x - 5)
        height = min(height, frame_height - y - 5)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        y_offset = y + pad_y + line_heights[0]
        for idx, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x + pad_x, y_offset),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            if idx + 1 < len(line_heights):
                y_offset += line_heights[idx + 1] + line_gap
        return frame, (x, y, width, height)

    def config_dict(self) -> Dict[str, float]:
        return asdict(self.config)
