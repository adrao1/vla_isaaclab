#!/bin/bash
set -e
source "$(dirname "$0")/activate.sh"
check_install_environment
cd "$YCB_PROJECT"
exec python scripts/run_scenario.py --headless --steps 300 "$@"
