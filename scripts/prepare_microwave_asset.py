#!/usr/bin/env python3
"""Convert the vendored furniture_sim microwave MJCF to USD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "assets/furniture_sim/microwave.xml"
OUTPUT = PROJECT / "assets/furniture_sim/generated/microwave.usd"
MANIFEST = PROJECT / "assets/furniture_sim/manifest.json"


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.experience = str(PROJECT / "configs/ycb.python.headless.kit")
app = AppLauncher(args).app

from isaacsim.core.utils.extensions import enable_extension
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
from pxr import Usd, UsdPhysics

from usd_asset_paths import relativize_project_paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    enable_extension("isaacsim.asset.importer.mjcf")
    converter = MjcfConverter(
        MjcfConverterCfg(
            asset_path=str(SOURCE),
            usd_dir=str(OUTPUT.parent),
            usd_file_name=OUTPUT.name,
            fix_base=True,
            import_sites=True,
            import_inertia_tensor=True,
            self_collision=False,
            make_instanceable=False,
            force_usd_conversion=True,
        )
    )
    output = Path(converter.usd_path)
    if not output.is_file():
        raise RuntimeError(f"MJCF conversion did not create {output}")
    rewritten_paths = relativize_project_paths(output.parent, PROJECT)
    portable_paths = {
        layer: sorted({relative_path for _, relative_path in replacements})
        for layer, replacements in rewritten_paths.items()
    }
    # Isaac Sim 4.5's MJCF importer applies ArticulationRootAPI to two empty
    # organizational prims as well as the actual microwave root.  Isaac Lab
    # requires exactly one root below the configured asset path.  Remove only
    # those extra API markers; all upstream geometry, bodies, collision and
    # joint definitions remain in the generated layers.
    stage = Usd.Stage.Open(str(output))
    for path in ("/microwave/_body_0", "/microwave/worldBody"):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    # The showcase MJCF wraps the real appliance in an offset, unnamed body.
    # The importer turns that empty wrapper into a second rigid body and a
    # disjoint fixed joint.  Deactivate both generated artifacts; the actual
    # microroot, meshes, colliders and door joint remain unchanged.
    for path in ("/microwave/_body_0", "/microwave/joints/rootJoint__body_0"):
        prim = stage.GetPrimAtPath(path)
        if prim:
            prim.SetActive(False)
    stage.GetRootLayer().Save()
    del stage
    converter_config = output.parent / "config.yaml"
    if converter_config.is_file():
        config_text = converter_config.read_text()
        config_text = config_text.replace(str(PROJECT) + "/", "")
        converter_config.write_text(config_text)
    manifest = {
        "name": "furniture_sim microwave",
        "source": "https://github.com/vikashplus/furniture_sim",
        "source_commit": "c97995afb81c9e2d7325b0069f9abc9a2c74a2f0",
        "license": "Apache-2.0",
        "source_mjcf": str(SOURCE.relative_to(PROJECT)),
        "source_sha256": sha256(SOURCE),
        "generated_usd": str(output.relative_to(PROJECT)),
        "generated_sha256": sha256(output),
        "converter": "Isaac Sim 4.5 isaacsim.asset.importer.mjcf",
        "door_joint": "micro0joint",
        "door_joint_range_rad": [-2.094, 0.0],
        "articulation_root": "/microwave/microroot",
        "relative_asset_paths": portable_paths,
        "postprocess": (
            "Removed duplicate ArticulationRootAPI markers and deactivated the empty showcase wrapper "
            "body/fixed joint emitted by the importer. Rewrote project-local USD dependencies as relative paths."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
