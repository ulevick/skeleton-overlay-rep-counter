import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from .pose_analyzer import AnalyzerConfig, PoseAnalyzer

MODEL_URLS = {
    "lite": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "full": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/1/pose_landmarker_full.task"
    ),
    "heavy": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    ),
}

OUTPUT_SUFFIXES = {
    "video": "annotated.mp4",
    "csv": "metrics.csv",
    "json": "metrics.json",
    "plot": "angles.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose-based exercise rep counter.")
    parser.add_argument("--input", help="Path to input video.")
    parser.add_argument("--input-dir", help="Process all videos in a directory.")
    parser.add_argument(
        "--extensions",
        default=".mp4,.mov,.avi,.mkv",
        help="Comma-separated extensions for input-dir.",
    )
    parser.add_argument("--output-dir", default="results", help="Directory for outputs.")
    parser.add_argument(
        "--output-video",
        default=None,
        help="Output video filename (default: <input_stem>_annotated.mp4; ignored in batch mode).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV filename (default: <input_stem>_metrics.csv; ignored in batch mode).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Output JSON filename (default: <input_stem>_metrics.json; ignored in batch mode).",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Output plot filename (default: <input_stem>_angles.png; ignored in batch mode).",
    )
    parser.add_argument("--hip-down", type=float, default=110.0, help="Hip angle threshold for down position.")
    parser.add_argument("--hip-up", type=float, default=160.0, help="Hip angle threshold for up position.")
    parser.add_argument("--knee-warn", type=float, default=150.0, help="Warn if knee angle below this.")
    parser.add_argument("--back-warn", type=float, default=160.0, help="Warn if back angle below this.")
    parser.add_argument(
        "--warning-hip-angle",
        type=float,
        default=150.0,
        help="Only emit technique warnings when hip angle is below this.",
    )
    parser.add_argument(
        "--warning-hold-frames",
        type=int,
        default=5,
        help="Frames required before a warning appears.",
    )
    parser.add_argument(
        "--exercise",
        choices=["auto", "rdl", "trx_pike"],
        default="auto",
        help="Exercise mode. Use auto to detect RDL vs TRX Pike.",
    )
    parser.add_argument("--exercise-frames", type=int, default=45, help="Frames used for auto detection.")
    parser.add_argument("--exercise-min-confidence", type=float, default=0.65, help="Auto detection confidence.")
    parser.add_argument(
        "--strap-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TRX strap edge detection.",
    )
    parser.add_argument("--strap-scale", type=float, default=0.5, help="Scale for strap detection.")
    parser.add_argument("--trx-pike-up", type=float, default=110.0, help="TRX Pike top threshold.")
    parser.add_argument("--trx-pike-down", type=float, default=160.0, help="TRX Pike plank threshold.")
    parser.add_argument("--trx-pike-up-lift", type=float, default=0.35, help="TRX Pike hip lift up threshold.")
    parser.add_argument("--trx-pike-down-lift", type=float, default=0.1, help="TRX Pike hip lift down threshold.")
    parser.add_argument("--trx-pike-stage-frames", type=int, default=5, help="Frames required to switch Pike stages.")
    parser.add_argument("--trx-pike-warn", type=float, default=150.0, help="Warn when body angle below this.")
    parser.add_argument("--trx-elbow-warn", type=float, default=160.0, help="Warn if elbows bend below this.")
    parser.add_argument("--trx-knee-warn", type=float, default=160.0, help="Warn if knees bend below this.")
    parser.add_argument("--smoothing-window", type=int, default=7, help="Smoothing window for angles.")
    parser.add_argument("--visibility-threshold", type=float, default=0.5, help="Min tracking quality.")
    parser.add_argument("--min-rep-interval", type=float, default=0.6, help="Min seconds between reps.")
    parser.add_argument("--model-complexity", type=int, default=2, choices=[0, 1, 2], help="MediaPipe Pose complexity.")
    parser.add_argument("--model-type", choices=["lite", "full", "heavy"], default="full", help="Tasks model type.")
    parser.add_argument(
        "--panel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render info panel on the right side.",
    )
    parser.add_argument("--panel-width", type=int, default=420, help="Panel width in pixels.")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec (e.g. mp4v, avc1, XVID).")
    parser.add_argument(
        "--model",
        default=None,
        help="Pose Landmarker model path (used when MediaPipe solutions API is unavailable).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def uses_tasks_api() -> bool:
    return not hasattr(mp, "solutions")


def ensure_model(path: str, model_type: str) -> str:
    if os.path.exists(path):
        return path
    parent = os.path.dirname(path) or "."
    ensure_dir(parent)
    print(f"Downloading model to {path} ...")
    urlretrieve(MODEL_URLS[model_type], path)
    return path


def save_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)


def save_plot(path: str, rows: List[Dict[str, object]], rep_events: List[float]) -> None:
    times = [row["timestamp_s"] for row in rows]
    hip_angles = [
        row["hip_angle_smooth"] if row["hip_angle_smooth"] is not None else np.nan for row in rows
    ]

    plt.figure(figsize=(12, 5))
    plt.plot(times, hip_angles, label="Hip angle (smoothed)")
    for event_time in rep_events:
        plt.axvline(event_time, color="tab:green", linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Romanian Deadlift Hip Angle Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def build_coaching_notes(
    rows: List[Dict[str, object]], summary: Dict[str, object], exercise: str
) -> str:
    total_frames = len(rows)
    if total_frames == 0:
        return "No frames were processed, so no coaching notes are available.\n"

    warning_counts: Dict[str, int] = {}
    tracking_low = 0
    for row in rows:
        if not row.get("tracking_ok", False):
            tracking_low += 1
        warnings = row.get("warnings") or ""
        for warning in [part.strip() for part in str(warnings).split(";") if part.strip()]:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1

    def ratio(count: int) -> float:
        return count / total_frames if total_frames else 0.0

    def warning_level(count: int) -> str:
        warning_ratio = ratio(count)
        if warning_ratio < 0.03:
            return "low"
        if warning_ratio < 0.1:
            return "medium"
        return "high"

    def push_warning(
        advice_list: List[str],
        count: int,
        low_msg: str,
        medium_msg: str,
        high_msg: str,
    ) -> None:
        if count <= 0:
            return
        level = warning_level(count)
        if level == "low":
            advice_list.append(low_msg)
        elif level == "medium":
            advice_list.append(medium_msg)
        else:
            advice_list.append(high_msg)

    lines: List[str] = []
    lines.append("Coaching notes")
    lines.append("=" * 14)
    lines.append(f"Exercise: {exercise.replace('_', ' ').title()}")
    lines.append(f"Reps counted: {summary.get('total_reps', 0)}")
    lines.append("")

    advice: List[str] = []
    warning_levels: List[str] = []
    if exercise == "rdl":
        back_count = warning_counts.get("Back rounded", 0)
        knee_count = warning_counts.get("Knees too bent", 0)
        if back_count:
            warning_levels.append(warning_level(back_count))
            push_warning(
                advice,
                back_count,
                "Occasional back rounding detected; keep a neutral spine and shorten the range if needed.",
                "Some reps showed back rounding; slow down and keep the chest up while hinging.",
                "Back rounding was frequent; reduce load or range and prioritize a neutral spine.",
            )
        if knee_count:
            warning_levels.append(warning_level(knee_count))
            push_warning(
                advice,
                knee_count,
                "Knee bend appeared on a few reps; keep only a soft bend and push the hips back.",
                "Several reps looked squatty; keep knees softer and hinge more at the hips.",
                "Knee bend was frequent; reset the hinge pattern and limit depth.",
            )
        if not advice:
            advice.append("Good consistency on back and knee angles. Keep the same hip-hinge pattern.")
    elif exercise == "trx_pike":
        elbow_count = warning_counts.get("Elbows bent", 0)
        knee_count = warning_counts.get("Knees slightly bent", 0)
        if elbow_count:
            warning_levels.append(warning_level(elbow_count))
            push_warning(
                advice,
                elbow_count,
                "A few frames showed elbow bend; keep arms long and press the floor away.",
                "Elbows bent on some reps; focus on locked arms and stable shoulders.",
                "Elbow bend was frequent; reduce range and keep arms straight.",
            )
        if knee_count:
            warning_levels.append(warning_level(knee_count))
            push_warning(
                advice,
                knee_count,
                "Minor knee bend showed up occasionally; aim for longer legs through the movement.",
                "Knee bend appeared on some reps; shorten the range until legs stay straighter.",
                "Knee bend was frequent; limit range and keep legs extended.",
            )
        if not advice:
            advice.append("Solid form cues. Keep the shoulders stable and the hips lifting smoothly.")
    else:
        advice.append("Exercise type not locked; re-run with --exercise for more specific feedback.")

    if warning_levels and all(level == "low" for level in warning_levels):
        advice.insert(0, "Overall form looked solid; only small touch-ups are needed.")

    tracking_ratio = ratio(tracking_low)
    if tracking_low and tracking_ratio >= 0.2:
        advice.append(
            "Tracking was unstable in several frames. Use a clear side view, good lighting, and keep the full body in frame."
        )
    elif tracking_low and tracking_ratio >= 0.08:
        advice.append(
            "Some frames had low tracking. A clearer side view and steadier framing will help."
        )

    lines.append("What to focus on next time:")
    for item in advice:
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def build_info_panel(height: int, width: int, record: Dict[str, object]) -> np.ndarray:
    panel = np.full((height, width, 3), (24, 24, 28), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    header_color = (255, 255, 255)
    text_color = (230, 230, 230)
    warning_color = (80, 80, 255)

    x = 20
    y = 40
    line_gap = 30

    def put_line(text: str, color=text_color, scale=0.65, thickness=2) -> None:
        nonlocal y
        cv2.putText(panel, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        y += line_gap

    exercise = record.get("exercise", "detecting")
    locked = record.get("exercise_locked", False)
    confidence = float(record.get("exercise_confidence", 0.0))
    if not locked:
        if exercise and exercise != "detecting":
            header = f"Detecting ({exercise.replace('_', ' ').title()})"
        else:
            header = "Detecting exercise"
    else:
        header = exercise.replace("_", " ").title()

    put_line(header, header_color, 0.7, 2)
    put_line(f"Confidence: {confidence:.2f}", text_color, 0.55, 1)
    y += 6
    put_line(f"Reps: {record['rep_count']}")
    if exercise == "trx_pike":
        put_line(f"Elbow angle: {record['elbow_angle_display']}")
        put_line(f"Body angle: {record['body_angle_display']}")
        put_line(f"Knee angle: {record['knee_angle_display']}")
    else:
        put_line(f"Hip angle: {record['hip_angle_display']}")
        put_line(f"Knee angle: {record['knee_angle_display']}")
        put_line(f"Back angle: {record['back_angle_display']}")
    put_line(f"Stage: {record['stage']}")
    put_line(f"Tracking: {record['tracking_display']}")

    warnings = record.get("warnings") or []
    if warnings:
        y += 12
        put_line("Warning:", warning_color, 0.7, 2)
        for warning in warnings:
            put_line(warning, warning_color, 0.65, 2)

    return panel


def resolve_inputs(args: argparse.Namespace) -> Tuple[List[Path], bool]:
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise RuntimeError(f"Input directory not found: {args.input_dir}")
        extensions = [
            ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
            for ext in args.extensions.split(",")
            if ext.strip()
        ]
        files = [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ]
        if not files:
            raise RuntimeError(f"No video files found in {args.input_dir} with {extensions}")
        return sorted(files), True

    if args.input:
        return [Path(args.input)], False

    raise RuntimeError("Provide --input or --input-dir.")


def build_output_paths(
    output_dir: Path, input_path: Path, batch_mode: bool, args: argparse.Namespace
) -> Dict[str, Path]:
    output_dir = output_dir / input_path.stem
    ensure_dir(str(output_dir))

    def pick_name(key: str, override: Optional[str]) -> str:
        if override and not batch_mode:
            return override
        return OUTPUT_SUFFIXES[key]

    return {
        "video": output_dir / pick_name("video", args.output_video),
        "csv": output_dir / pick_name("csv", args.output_csv),
        "json": output_dir / pick_name("json", args.output_json),
        "plot": output_dir / pick_name("plot", args.output_plot),
        "summary": output_dir / "summary.json",
        "notes": output_dir / "coaching_notes.txt",
    }


def process_video(
    input_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    config: AnalyzerConfig,
    batch_mode: bool,
) -> Dict[str, object]:
    analyzer = PoseAnalyzer(config=config)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        analyzer.close()
        raise RuntimeError(f"Unable to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    outputs = build_output_paths(output_dir, input_path, batch_mode, args)
    output_width = width + args.panel_width if args.panel else width
    output_height = height
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(outputs["video"]), fourcc, fps, (output_width, output_height))

    rows: List[Dict[str, object]] = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_s = frame_index / fps
        annotated, record = analyzer.process_frame(frame, timestamp_s, with_overlay=not args.panel)
        record["frame_index"] = frame_index
        record["timestamp_s"] = round(timestamp_s, 4)
        warnings_text = "; ".join(record["warnings"]) if record["warnings"] else ""
        row = record.copy()
        row["warnings"] = warnings_text
        rows.append(row)
        if args.panel:
            panel = build_info_panel(height, args.panel_width, record)
            combined = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            combined[:, :width] = annotated
            combined[:, width:] = panel
            writer.write(combined)
        else:
            writer.write(annotated)
        frame_index += 1

    cap.release()
    writer.release()
    analyzer.close()

    save_csv(str(outputs["csv"]), rows)
    save_json(str(outputs["json"]), rows)
    save_plot(str(outputs["plot"]), rows, analyzer.rep_events)

    summary = {
        "input": str(input_path),
        "output_video": str(outputs["video"]),
        "output_csv": str(outputs["csv"]),
        "output_json": str(outputs["json"]),
        "output_plot": str(outputs["plot"]),
        "output_notes": str(outputs["notes"]),
        "frames": frame_index,
        "fps": fps,
        "total_reps": analyzer.rep_count,
        "exercise_mode": args.exercise,
        "detected_exercise": analyzer.exercise_type,
        "config": analyzer.config_dict(),
        "generated_at": datetime.now().isoformat(),
    }
    with open(outputs["summary"], "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    exercise_label = analyzer.exercise_type or "detecting"
    notes_text = build_coaching_notes(rows, summary, exercise_label)
    with open(outputs["notes"], "w", encoding="utf-8") as file:
        file.write(notes_text)

    print(f"Done. Reps: {analyzer.rep_count}")
    print(f"Video: {outputs['video']}")
    print(f"CSV:   {outputs['csv']}")
    print(f"JSON:  {outputs['json']}")
    print(f"Plot:  {outputs['plot']}")
    print(f"Notes: {outputs['notes']}")

    return summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_path = None
    if uses_tasks_api():
        model_path = args.model
        if not model_path:
            model_path = os.path.join("data", "models", f"pose_landmarker_{args.model_type}.task")
        model_path = ensure_model(model_path, args.model_type)

    config = AnalyzerConfig(
        hip_down_angle=args.hip_down,
        hip_up_angle=args.hip_up,
        knee_warning_angle=args.knee_warn,
        back_warning_angle=args.back_warn,
        warning_hip_angle=args.warning_hip_angle,
        warning_hold_frames=args.warning_hold_frames,
        exercise_mode=args.exercise,
        exercise_classify_frames=args.exercise_frames,
        exercise_min_confidence=args.exercise_min_confidence,
        strap_detection=args.strap_detection,
        strap_detection_scale=args.strap_scale,
        trx_pike_up_angle=args.trx_pike_up,
        trx_pike_down_angle=args.trx_pike_down,
        trx_pike_up_lift=args.trx_pike_up_lift,
        trx_pike_down_lift=args.trx_pike_down_lift,
        trx_pike_stage_frames=args.trx_pike_stage_frames,
        trx_pike_warning_angle=args.trx_pike_warn,
        trx_pike_elbow_warning_angle=args.trx_elbow_warn,
        trx_pike_knee_warning_angle=args.trx_knee_warn,
        visibility_threshold=args.visibility_threshold,
        smoothing_window=args.smoothing_window,
        min_rep_interval_s=args.min_rep_interval,
        model_complexity=args.model_complexity,
        model_path=model_path,
    )
    inputs, batch_mode = resolve_inputs(args)
    output_dir = Path(args.output_dir)

    summaries: List[Dict[str, object]] = []
    for input_path in inputs:
        summaries.append(process_video(input_path, output_dir, args, config, batch_mode))

    if batch_mode:
        ensure_dir(str(output_dir))
        batch_summary_path = output_dir / "batch_summary.json"
        with open(batch_summary_path, "w", encoding="utf-8") as file:
            json.dump(summaries, file, indent=2)
        print(f"Batch summary: {batch_summary_path}")


if __name__ == "__main__":
    main()
