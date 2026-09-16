#!/bin/bash
set -e
source /home/vlakbnn/jiajunl4/YCB_Object/scripts/activate.sh
check_install_environment
cd "$YCB_PROJECT"
exec python scripts/run_scenario.py "$@"
