#!/usr/bin/env python3
"""Inspect articulation, rigid-body, collision, and joint schemas in the microwave USD."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT = Path(__file__).resolve().parents[1]
ASSET = PROJECT / "assets/furniture_sim/generated/microwave.usd"

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.experience = str(PROJECT / "configs/ycb.python.headless.kit")
app = AppLauncher(args).app

from pxr import Usd, UsdPhysics


def main() -> None:
    stage = Usd.Stage.Open(str(ASSET))
    print(f"default_prim={stage.GetDefaultPrim().GetPath()}", flush=True)
    for prim in stage.Traverse():
        interesting = []
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            interesting.append("articulation_root")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid = UsdPhysics.RigidBodyAPI(prim)
            interesting.append(
                f"rigid_body enabled={rigid.GetRigidBodyEnabledAttr().Get()} "
                f"kinematic={rigid.GetKinematicEnabledAttr().Get()}"
            )
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            interesting.append("collision")
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            interesting.append(
                f"revolute lower={joint.GetLowerLimitAttr().Get()} upper={joint.GetUpperLimitAttr().Get()} "
                f"axis={joint.GetAxisAttr().Get()}"
            )
            interesting.append(
                f"body0={joint.GetBody0Rel().GetTargets()} body1={joint.GetBody1Rel().GetTargets()} "
                f"enabled={joint.GetJointEnabledAttr().Get()}"
            )
        if interesting:
            print(f"{prim.GetPath()} [{', '.join(interesting)}]", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
