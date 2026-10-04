# AI

Catfish (lele) pond monitoring with YOLOv8 (ultralytics) + OpenCV. Standalone scripts, no package layout yet.

## Files
- `train.py` — trains `yolov8n.pt` on `archive/catfish.yaml` (1 class: `catfish`), outputs to `runs/detect/lele_model_v3/`. Then runs a prediction on `lele.mp4`.
- `index.py` — current demo: loops `lele.mp4`, applies a water-colour filter, runs `runs/detect/lele_model_v3/weights/best.pt`, draws a HUD, shows an OpenCV window. Telemetry (pH, temp, turbidity, TDS) is **simulated** by time-based phases (normal -> keruh -> bersih), not read from real sensors.
- `index_old.py` — previous version, reference only.
- `archive/` dataset (note: `catfish.yaml` has an absolute `path: D:/Python/Hackton/archive` that doesn't match the current location — fix before retraining).
- `Dataset.xlsx`, `yolov8n.pt`, `lele.mp4`, `runs/` — large assets, don't edit or commit.

## main_engine.py (entry point for the full pipeline)
Refactor of `index.py`: YOLOv8 + water-colour filter + analytics, POSTs to `hackton-be`.
- `python main_engine.py` (window) / `--headless` / `--max-seconds N`. Run from this folder. Needs the backend on `BACKEND_URL` (default `http://127.0.0.1:8000`).
- `engine/analytics.py` — `TelemetryAnalytics` (10-sample rolling window, 1 sample per video second; `get_trends()` returns `LONJAKAN_AMONIA` if pH delta > 0.5 or temp delta > 1.0), `calculate_fcr`, `calculate_savings` (15% of feed x Rp 12.000/kg).
- Every `POST_INTERVAL` s (default 2) -> `POST /api/telemetry/` with the **model field names**: `catfish_count, ph, temperature, turbidity, tds, status, recommendation`. `status` must start with `IDEAL` / `PERINGATAN` / `SANGAT BAIK` (backend derives `level` from it; max 50 chars).
- On each water-mode change -> `POST /api/biomass/` (demo constants `DEMO_*`: feed, fish count, biomass gain are simulated).
- POSTs run in a one-worker thread pool so inference never blocks on the network.
- Telegram alerts are intentionally not part of this project.

## Notes
- Inference is the bottleneck: resize before `predict`, JPEG-encode at quality ~70 for the stream.
- Run scripts from this folder (relative paths to `runs/`, `lele.mp4`).
- UI strings and comments are in Indonesian; keep that style.
