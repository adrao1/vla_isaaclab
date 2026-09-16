"""Cache selected files from the official Isaac Sim 4.5 asset library."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

from pxr import Sdf, UsdUtils


SOURCE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/"
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024


def prepare_official_assets(project: Path, assets: list[dict]) -> dict[str, Path]:
    """Download selected official assets and every relative dependency they reference."""
    root = project / "assets" / "Isaac" / "4.5"
    root.mkdir(parents=True, exist_ok=True)
    visited: set[str] = set()
    manifest: list[dict] = []
    downloaded_bytes = 0

    def fetch(url: str) -> Path:
        nonlocal downloaded_bytes
        if not url.startswith(SOURCE):
            raise RuntimeError(f"Dependency outside the Isaac Sim 4.5 asset root: {url}")

        relative = url[len(SOURCE) :]
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe asset path: {relative}")
        if url in visited:
            return target
        visited.add(url)

        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            print(f"[ASSET] Downloading {url}", flush=True)
            with urlopen(url, timeout=60) as response:
                data = response.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise RuntimeError(f"Asset is unexpectedly larger than 100 MiB: {url}")
            downloaded_bytes += len(data)
            if downloaded_bytes > MAX_DOWNLOAD_BYTES:
                raise RuntimeError("Dinnerware download unexpectedly exceeded 500 MiB; stopping.")
            with target.open("xb") as stream:
                stream.write(data)

        manifest.append(
            {
                "path": str(target.relative_to(project)),
                "source": url,
                "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )

        if target.suffix.lower() in (".usd", ".usda", ".usdc"):
            layer = Sdf.Layer.FindOrOpen(str(target))
            if layer is None:
                raise RuntimeError(f"Cannot read USD: {target}")
            dependencies: list[str] = []

            def collect(path: str) -> str:
                dependencies.append(path)
                return path

            UsdUtils.ModifyAssetPaths(layer, collect)
            for dependency in dependencies:
                if not dependency or (dependency.endswith(".mdl") and "/" not in dependency):
                    continue
                fetch(urljoin(url, dependency))
        return target

    result = {}
    for asset in assets:
        result[asset["name"]] = fetch(SOURCE + asset["asset"])

    (root.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result
