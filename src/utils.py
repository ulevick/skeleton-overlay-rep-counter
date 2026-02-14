"""Geometry and helper utilities."""

import math
from typing import Iterable, Tuple

import numpy as np


def calculate_angle(point_a: Iterable[float], point_b: Iterable[float], point_c: Iterable[float]) -> float:
    """Return angle ABC in degrees for three 2D points."""
    a = np.array(point_a, dtype=np.float32)
    b = np.array(point_b, dtype=np.float32)
    c = np.array(point_c, dtype=np.float32)

    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0.0 or norm_bc == 0.0:
        return float("nan")

    cosine_angle = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cosine_angle = float(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine_angle)))


def safe_mean(values: Iterable[float]) -> float:
    numeric_values = [value for value in values if value is not None and not math.isnan(value)]
    if not numeric_values:
        return float("nan")
    return float(np.mean(numeric_values))


def visibility_score(landmarks, indices: Iterable[int]) -> float:
    visibilities = []
    for index in indices:
        landmark = landmarks[index]
        visibility = getattr(landmark, "visibility", None)
        if visibility is None:
            visibility = getattr(landmark, "presence", 1.0)
        visibilities.append(float(visibility))
    return float(np.mean(visibilities))


def landmark_xy(landmarks, index: int) -> Tuple[float, float]:
    landmark = landmarks[index]
    return float(landmark.x), float(landmark.y)


def landmark_xy_visibility(landmarks, index: int) -> Tuple[Tuple[float, float], float]:
    landmark = landmarks[index]
    visibility = getattr(landmark, "visibility", None)
    if visibility is None:
        visibility = getattr(landmark, "presence", 1.0)
    return (float(landmark.x), float(landmark.y)), float(visibility)
