#!/usr/bin/env bash
# Creates .venv with pinned versions. SAM2 is installed without build isolation (avoids pip pulling a second torch)
# and without its optional CUDA extension (only used for small-hole filling in mask postprocessing).
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q pip==24.3.1
grep -v '^sam2' requirements.txt > .req_nosam.txt
.venv/bin/pip install -r .req_nosam.txt setuptools==75.6.0 wheel==0.45.1
SAM2_BUILD_CUDA=0 .venv/bin/pip install --no-build-isolation --no-deps sam2==1.1.0
rm .req_nosam.txt
.venv/bin/python -c "import torch,sam2,ultralytics,transformers,numpy,cv2;print('OK', torch.__version__, numpy.__version__, ultralytics.__version__, transformers.__version__, cv2.__version__, torch.cuda.is_available())"
