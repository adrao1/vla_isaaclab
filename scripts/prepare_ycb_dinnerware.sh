#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/activate.sh"
check_install_environment
cd "$ISAAC_SIMLAB_PROJECT"

python scripts/prepare_ycb_dinnerware.py

python IsaacLab/scripts/tools/convert_mesh.py \
  assets/YCB/dinnerware/024_bowl/visual.obj \
  assets/YCB/dinnerware/024_bowl/024_bowl_physics.usd \
  --headless --experience configs/ycb.python.headless.kit \
  --collision-approximation convexDecomposition --mass 0.147

python IsaacLab/scripts/tools/convert_mesh.py \
  assets/YCB/dinnerware/029_plate/visual.obj \
  assets/YCB/dinnerware/029_plate/029_plate_physics.usd \
  --headless --experience configs/ycb.python.headless.kit \
  --collision-approximation convexDecomposition --mass 0.279
