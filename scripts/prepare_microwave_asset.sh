#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
source scripts/activate.sh
check_install_environment
python scripts/prepare_microwave_asset.py
