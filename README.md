# Exercise Rep Counter (Pose-based)

Pose-based video analyzer that counts reps and produces annotated outputs.

Supported exercises:
- Romanian Deadlift (RDL)
- TRX Pike

The CLI can auto-detect the exercise or force a specific mode.

## Project structure
- `src/run.py` CLI entry point and batch processing
- `src/pose_analyzer.py` pose detection, rep counting, warnings, overlays
- `src/utils.py` geometry helpers
- `data/` sample videos and MediaPipe models
- `results/` sample outputs for the provided videos
- `requirements.txt` Python dependencies

## Setup
- Python 3.9+
- Install deps:
  - `pip install -r requirements.txt`

## Run
Run from the repo root (important for package imports):

- Single video:
  - `python -m src.run --input data/rdl_sample.mp4`
- Batch mode (all videos in a folder):
  - `python -m src.run --input-dir data`
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
- RDL reps are counted by hip-angle stage transitions (`--hip-down` / `--hip-up`) with
  a minimum time between reps.
- Warnings are emitted only when tracking is good and thresholds are held for N frames.
- Coaching notes are derived from warning frequency and tracking quality.

## Outputs
Outputs go to `results/` by default (change with `--output-dir`). Each input video gets its own
folder so results are easy to find:
- `results/<input_stem>/annotated.mp4`
- `results/<input_stem>/metrics.csv`
- `results/<input_stem>/metrics.json`
- `results/<input_stem>/angles.png`
- `results/<input_stem>/summary.json`
- `results/<input_stem>/coaching_notes.txt` (plain-text recommendations)

You can override `--output-video/--output-csv/--output-json/--output-plot` for single-video runs,
but the files still land in `results/<input_stem>/`. Batch mode always uses the default filenames.

Outputs include pose overlays/warnings, per-frame metrics, a hip angle plot with rep markers, and
plain-text coaching notes. Batch runs also create `results/batch_summary.json`.

## Key CLI options
Run `python -m src.run --help` for the full list. Common options:
- `--exercise auto|rdl|trx_pike`
- `--hip-down`, `--hip-up`, `--knee-warn`, `--back-warn`
- `--trx-pike-up`, `--trx-pike-down`, `--trx-pike-up-lift`, `--trx-pike-down-lift`
- `--warning-hold-frames`, `--min-rep-interval`, `--smoothing-window`, `--visibility-threshold`
- `--panel/--no-panel`, `--panel-width`
- `--model-type lite|full|heavy`, `--model`

## MediaPipe models
If the MediaPipe Solutions API is unavailable, the code switches to the Tasks API and needs a
`.task` model file. This repo already includes:
- `data/models/pose_landmarker_lite.task`
- `data/models/pose_landmarker_full.task`
- `data/models/pose_landmarker_heavy.task`

If a model file is missing, it is downloaded automatically to
`data/models/pose_landmarker_<type>.task`.

## Known limitations
- Accuracy depends on visibility and camera angle (side view works best).
- Occlusions or low light reduce tracking quality and may suppress warnings.
- Only RDL and TRX Pike are supported by the current heuristics.
