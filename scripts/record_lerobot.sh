#!/bin/bash
set -e
source "$(dirname "$0")/activate.sh"
check_install_environment
cd "$YCB_PROJECT"
exec python scripts/run_scenario.py \
  --headless \
  --record-format lerobot \
  --episodes 2 \
  --steps 120 \
  --dataset-name g1_ycb_dinnerware_raise_lower \
  --task-prompt "Raise and lower both arms in front of the YCB bowl and plate." \
  "$@"
