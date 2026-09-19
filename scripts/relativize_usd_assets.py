#!/usr/bin/env python3
"""Rewrite generated project-local USD dependencies as relative asset paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("usd_root", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.experience = str(PROJECT / "configs/ycb.python.headless.kit")
app = AppLauncher(args).app

from usd_asset_paths import relativize_project_paths


if __name__ == "__main__":
    try:
        root = args.usd_root.resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        print(json.dumps(relativize_project_paths(root, PROJECT), indent=2))
    finally:
        app.close()
