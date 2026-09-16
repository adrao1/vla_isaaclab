#!/bin/bash
set -e
source /home/vlakbnn/jiajunl4/YCB_Object/scripts/activate.sh
check_install_environment
cd "$YCB_PROJECT"
exec python src/run_dinnerware_scene.py "$@"
