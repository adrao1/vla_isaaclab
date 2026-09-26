from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import (
    MeshConverter,
    MeshConverterCfg,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSET_ROOT = (
    PROJECT_ROOT
    / "assets"
    / "xsim"
    / "kitchen_env"
)


def convert(
    source: Path,
    output_dir: Path,
    output_name: str,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = MeshConverterCfg(
        asset_path=str(source),
        usd_dir=str(output_dir),
        usd_file_name=output_name,
        force_usd_conversion=True,
        make_instanceable=False,
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    )

    converter = MeshConverter(cfg)

    print(f"{source}")
    print(f"  -> {converter.usd_path}")


def main():
    convert(
        ASSET_ROOT / "Kitchen.obj",
        ASSET_ROOT / "usd",
        "Kitchen.usd",
    )

    convert(
        ASSET_ROOT
        / "mustard"
        / "mesh"
        / "mustard.obj",
        ASSET_ROOT
        / "mustard"
        / "usd",
        "mustard.usd",
    )


if __name__ == "__main__":
    main()
