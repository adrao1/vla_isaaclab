#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/activate.sh"
check_install_environment
cd "$VLA_ISAACLAB_PROJECT"

python scripts/prepare_ycb_dinnerware.py

python "$ISAACLAB_ROOT/scripts/tools/convert_mesh.py" \
  assets/YCB/dinnerware/024_bowl/visual.obj \
  assets/YCB/dinnerware/024_bowl/024_bowl_physics.usd \
  --headless --experience "$VLA_ISAACLAB_PROJECT/configs/ycb.python.headless.kit" \
  --collision-approximation convexDecomposition --mass 0.147

python "$ISAACLAB_ROOT/scripts/tools/convert_mesh.py" \
  assets/YCB/dinnerware/029_plate/visual.obj \
  assets/YCB/dinnerware/029_plate/029_plate_physics.usd \
  --headless --experience "$VLA_ISAACLAB_PROJECT/configs/ycb.python.headless.kit" \
  --collision-approximation convexDecomposition --mass 0.279

python scripts/relativize_usd_assets.py assets/YCB/dinnerware --headless
