#!/usr/bin/env python3
"""Download and normalize the official YCB bowl and plate meshes."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PROJECT / "assets/YCB/original"
MODEL_ROOT = PROJECT / "assets/YCB/dinnerware"

ASSETS = {
    "024_bowl": {
        "url": "https://ycb-benchmarks.s3.amazonaws.com/data/google/024_bowl_google_16k.tgz",
        "archive_sha256": "ac73b566d4f17c3cd5956684875c3a20dd4c8777821df62f5fee81f6b7751868",
        "mass_kg": 0.147,
        "published_dimensions_m": [0.159, 0.159, 0.053],
    },
    "029_plate": {
        "url": "https://ycb-benchmarks.s3.amazonaws.com/data/google/029_plate_google_16k.tgz",
        "archive_sha256": "1604d51aa6b7950a1491e7ffab0f8336a625d0a109e4cfb33b7debc94d670255",
        "mass_kg": 0.279,
        "published_dimensions_m": [0.258, 0.258, 0.024],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(spec: dict, archive: Path) -> None:
    if archive.is_file() and sha256(archive) == spec["archive_sha256"]:
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".part")
    urllib.request.urlretrieve(spec["url"], temporary)
    if sha256(temporary) != spec["archive_sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {spec['url']}")
    temporary.replace(archive)


def extract_member(bundle: tarfile.TarFile, member_name: str, destination: Path) -> None:
    member = bundle.getmember(member_name)
    source = bundle.extractfile(member)
    if source is None:
        raise RuntimeError(f"Missing archive member: {member_name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source, destination.open("wb") as output:
        shutil.copyfileobj(source, output)


def center_mesh(source: Path, destination: Path) -> list[float]:
    lines = source.read_text().splitlines()
    vertices = [tuple(map(float, line.split()[1:4])) for line in lines if line.startswith("v ")]
    lower = [min(vertex[i] for vertex in vertices) for i in range(3)]
    upper = [max(vertex[i] for vertex in vertices) for i in range(3)]
    offset = [-(lower[0] + upper[0]) / 2, -(lower[1] + upper[1]) / 2, -lower[2]]
    normalized = []
    for line in lines:
        if line.startswith("v "):
            fields = line.split()
            xyz = [float(fields[i + 1]) + offset[i] for i in range(3)]
            line = f"v {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}"
        normalized.append(line)
    destination.write_text("\n".join(normalized) + "\n")
    return [upper[i] - lower[i] for i in range(3)]


def main() -> None:
    manifest = {"license": "CC BY 4.0", "source": "YCB Object and Model Set", "assets": {}}
    for name, spec in ASSETS.items():
        archive = ARCHIVE_ROOT / f"{name}_google_16k.tgz"
        download(spec, archive)
        output = MODEL_ROOT / name
        output.mkdir(parents=True, exist_ok=True)
        prefix = f"{name}/google_16k/"
        with tarfile.open(archive, "r:gz") as bundle:
            for filename in ("textured.obj", "textured.mtl", "texture_map.png"):
                extract_member(bundle, prefix + filename, output / ("source.obj" if filename == "textured.obj" else filename))
        mesh_dimensions = center_mesh(output / "source.obj", output / "visual.obj")
        manifest["assets"][name] = {
            **spec,
            "archive": str(archive.relative_to(PROJECT)),
            "visual_mesh": str((output / "visual.obj").relative_to(PROJECT)),
            "mesh_dimensions_m": mesh_dimensions,
            "coordinate_normalization": "XY bounds centered; minimum Z moved to zero; no scaling",
        }
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    (MODEL_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
