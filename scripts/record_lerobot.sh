#!/bin/bash
set -e
source /home/vlakbnn/jiajunl4/YCB_Object/scripts/activate.sh
check_install_environment
cd "$YCB_PROJECT"
exec python scripts/run_scenario.py \
  --headless \
  --record-format lerobot \
  --episodes 2 \
  --steps 120 \
  --dataset-name g1_dinnerware_raise_lower \
  "$@"
