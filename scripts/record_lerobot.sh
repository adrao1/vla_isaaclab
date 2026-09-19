#!/bin/bash
set -e
source "$(dirname "$0")/activate.sh"
check_install_environment
cd "$VLA_ISAACLAB_PROJECT"
exec python scripts/run_scenario.py \
  --headless \
  --record-format lerobot \
  --episodes 2 \
  --steps 120 \
  --dataset-name vla_isaaclab_scene_preview \
  --task-prompt "Hold the Unitree G1 still while previewing the tabletop YCB scene." \
  "$@"
