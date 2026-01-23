#!/usr/bin/env bash
set -euo pipefail

# Assumes: fresh conda env already activated, and torch+torchvision already installed.

python -m pip install -U pip
pip install -r requirements.txt

# Build tooling (MMCV recommends ninja/psutil; pybind11/packaging help some builds)
pip install ninja psutil packaging pybind11

# 1) MMEngine
pip install "mmengine==0.10.7"

# 2) Build MMCV from source (with ops) to make sure it matches torch+cuda
pip uninstall -y mmcv mmcv-full || true

MMCV_TAG="v2.1.0"

rm -rf /tmp/mmcv-build
git clone --depth 1 --branch "${MMCV_TAG}" https://github.com/open-mmlab/mmcv.git /tmp/mmcv-build
cd /tmp/mmcv-build

if ! command -v nvcc >/dev/null 2>&1; then
  echo "❌ nvcc not found."
  echo "   Set CUDA_HOME to your CUDA toolkit (e.g. /usr/local/cuda) and ensure nvcc is on PATH."
  exit 1
fi

export MMCV_WITH_OPS=1

# Optional speed/repro: set arch list (A30 is sm_80)
# export TORCH_CUDA_ARCH_LIST="8.0"

pip install -v --no-build-isolation .

# Validate MMCV build (repo check)
python .dev_scripts/check_installation.py

# Validate the *installed* mmcv, not the source checkout
( cd / && PYTHONPATH= python - <<'PY'
import torch, mmcv, pathlib
print("mmcv file:", pathlib.Path(mmcv.__file__).resolve())
print("torch:", torch.__version__)
print("mmcv:", mmcv.__version__)
print("cuda available:", torch.cuda.is_available())
from mmcv.ops import roi_align
print("mmcv.ops roi_align import: OK")
PY
)

cd - >/dev/null

# 3) MMDetection
pip install "mmdet==3.3.0"

# Final check from neutral dir
cd /
PYTHONPATH= python - <<'PY'
import torch, mmcv, mmdet, pathlib
print("mmcv file:", pathlib.Path(mmcv.__file__).resolve())
print("torch:", torch.__version__)
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("cuda available:", torch.cuda.is_available())
from mmcv.ops import roi_align
print("mmcv.ops roi_align import: OK")
from mmdet.datasets import CocoDataset
print("CocoDataset import OK")
PY

echo "✅ Install complete"
