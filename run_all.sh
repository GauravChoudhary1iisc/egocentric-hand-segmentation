#!/usr/bin/env bash
# Full pipeline. Each step logs to logs/<step>.log (stdout+stderr) and writes its JSON/CSV into outputs/.
set -euo pipefail
cd "$(dirname "$0")"
export DATA_DIR="${DATA_DIR:-$PWD/data_raw/data}" PYTHONHASHSEED=42
PY=.venv/bin/python
mkdir -p logs outputs
run() { echo "=== $1 ==="; $PY "$1.py" 2>&1 | tee "logs/$1.log"; }
run s1_data_checks      # split.json, features.csv
run s2_val_baseline     # config.json (GDINO thresholds, bucket cut-offs), gate.json  -- val only
run s3_train_yolo       # training_log.csv, YOLO weights, yolo_conf in config.json  -- train/val only
run s4_evaluate         # per_image.csv, metrics.json, gate result, regressions, drift.json -- test, once
run s6_throughput       # throughput.json
run s6_docker_check     # docker build/run + mask comparison (or NOT DONE with reason)
run s7_face_blur        # optional, qualitative
run make_readme         # README.md from the JSON files
