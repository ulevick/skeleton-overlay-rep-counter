# Exercise Rep Counter (Pose-based)

Pose-based video analyzer that counts reps and produces annotated outputs.

Supported exercises:
- Romanian Deadlift (RDL)
- TRX Pike

The CLI can auto-detect the exercise or force a specific mode.

## Problem and motivation
Personal coaching can be expensive or unavailable, and beginners often struggle to verify their
technique when training alone. This project offers a lightweight way to review form from a video:
it overlays a pose skeleton, counts reps, and generates data-driven feedback so users can spot inconsistencies and improve safely over time.

## Project structure
- `src/run.py` CLI entry point and batch processing
- `src/pose_analyzer.py` pose detection and frame-level orchestration
- `src/exercise_detector.py` exercise auto-detection and confidence scoring
- `src/rep_counter.py` rep counting logic and warning thresholds
- `src/utils.py` geometry helpers
- `data/` sample videos (correct + incorrect form) and MediaPipe models
- `results/` sample outputs for the provided videos
- `requirements.txt` Python dependencies

## Setup
- Python 3.9+
- Install deps:
  - `pip install -r requirements.txt`

## Run
Run from the repo root (important for package imports):

- Batch mode (all videos in a folder):
  - `python -m src.run --input-dir data`
- Single video:
  - `python -m src.run --input data/example.mp4`
- Custom path example:
  - `python -m src.run --input "C:\path\to\video.mp4"`

## How it works (high level)
- MediaPipe Pose detects landmarks per frame (Solutions API if available, otherwise Tasks API).
- Angles are computed from landmarks:
  - Hip: shoulder–hip–knee
  - Knee: hip–knee–ankle
  - Back: hip–shoulder–head
  - Elbow: shoulder–elbow–wrist
  - Body: shoulder–hip–ankle
- Angles are smoothed with a sliding window before decisions are made.
- RDL reps are counted by hip-angle stage transitions, using `--hip-down` / `--hip-up`
  thresholds that are adjusted by the observed upright hip baseline (min range + return
  margin) and a minimum time between reps.
- TRX Pike reps use hip-lift + body-angle stages (plank - pike - plank) with minimum timing.
- Exercise type can be auto-detected using heuristic scores and a short voting window;
  TRX Pike can also lock early when plank-like signals and strap/hip-lift cues are present.
- Warnings are emitted only when tracking is good and thresholds are held for N frames.
- Coaching notes are derived from warning frequency and tracking quality.

## Outputs
By default, outputs are written to `results/<input_stem>/` (change root with `--output-dir`):
- `annotated.mp4` — video with pose overlay, rep counter, and warnings.
- `metrics.csv` — per-frame numeric metrics (angles, tracking, stage, warnings).
- `metrics.json` — same metrics as JSON for easier parsing.
- `angles.png` — hip angle plot with rep markers (most meaningful for RDL; still generated for other modes).
- `summary.json` — run summary (inputs, outputs, reps, config, timestamps).
- `coaching_notes.txt` — plain-text recommendations based on detected issues.

Batch runs also create `results/batch_summary.json`.

## Key CLI options
Run `python -m src.run --help` for the full list. Most commonly used:
- `--input` or `--input-dir` (choose a single video or a folder)
- `--exercise auto|rdl|trx_pike` (auto-detect or force a mode)
- `--output-dir` (where results are saved)

## MediaPipe models
If the MediaPipe Solutions API is unavailable, the code switches to the Tasks API and needs a
`.task` model file. If the file is missing, it is downloaded automatically. You can also
pre-place model files in `data/models/`:
- `data/models/pose_landmarker_lite.task`
- `data/models/pose_landmarker_full.task`
- `data/models/pose_landmarker_heavy.task`

If a model file is missing, it is downloaded automatically to
`data/models/pose_landmarker_<type>.task`.

## Known limitations
- Accuracy depends on visibility and camera angle (side view works best).
- Occlusions or low light reduce tracking quality and may suppress warnings.
- Only RDL and TRX Pike are supported by the current heuristics.
