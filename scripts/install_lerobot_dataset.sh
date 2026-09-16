#!/bin/bash
set -e

source /home/vlakbnn/jiajunl4/YCB_Object/scripts/activate.sh
cd "$YCB_PROJECT"

check_install_environment
python -m pip install -c configs/lerobot-constraints.txt \
  datasets==4.1.1 pyarrow==25.0.1 pandas==2.3.3 av==15.1.0 \
  jsonlines==4.0.0 accelerate==1.15.0

check_install_environment
python -m pip install --no-deps \
  --index-url https://download.pytorch.org/whl/cu121 \
  torchvision==0.20.1+cu121

check_install_environment
python -m pip install --no-deps lerobot==0.4.3

python - <<'PY'
from importlib.metadata import version

import numpy
import torch
import torchvision

print("Isaac Sim:", version("isaacsim"))
print("LeRobot:", version("lerobot"))
print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("NumPy:", numpy.__version__)
PY
