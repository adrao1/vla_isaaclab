"""Utilities for making generated USD dependencies portable."""

from __future__ import annotations

import os
from pathlib import Path

from pxr import Sdf, UsdUtils


def relativize_project_paths(usd_root: Path, project: Path) -> dict[str, list[tuple[str, str]]]:
    """Rewrite absolute project-local asset paths below ``usd_root`` as relative paths."""
    project = project.resolve()
    rewritten: dict[str, list[tuple[str, str]]] = {}

    for layer_path in sorted(usd_root.rglob("*.usd")):
        layer_path = layer_path.resolve()
        layer = Sdf.Layer.FindOrOpen(str(layer_path))
        if layer is None:
            raise RuntimeError(f"Cannot open generated USD layer: {layer_path}")

        changes: list[tuple[str, str]] = []

        def make_relative(asset_path: str) -> str:
            if not asset_path:
                return asset_path
            candidate = Path(asset_path)
            if not candidate.is_absolute():
                return asset_path
            try:
                candidate.relative_to(project)
            except ValueError:
                return asset_path

            relative = Path(os.path.relpath(candidate, start=layer_path.parent)).as_posix()
            if not relative.startswith("."):
                relative = f"./{relative}"
            changes.append((asset_path, relative))
            return relative

        UsdUtils.ModifyAssetPaths(layer, make_relative)
        layer.subLayerPaths = [make_relative(path) for path in layer.subLayerPaths]
        temporary = layer_path.with_name(f".{layer_path.stem}.portable.usd")
        try:
            if not layer.Export(str(temporary)):
                raise RuntimeError(f"Cannot export portable USD layer: {layer_path}")
            os.replace(temporary, layer_path)
        finally:
            temporary.unlink(missing_ok=True)
        if changes:
            rewritten[str(layer_path.relative_to(project))] = changes

    return rewritten
