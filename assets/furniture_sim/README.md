# furniture_sim microwave asset

This directory vendors the existing articulated microwave model from
[`vikashplus/furniture_sim`](https://github.com/vikashplus/furniture_sim) at
commit `c97995afb81c9e2d7325b0069f9abc9a2c74a2f0`.

- Upstream author: Vikash Kumar
- Upstream license: Apache License 2.0 (see `LICENSE`)
- Source format: MuJoCo MJCF
- Door joint: `micro0joint`, range `[-2.094, 0]` radians
- Generated USD: `generated/microwave.usd`, produced with the Isaac Sim 4.5
  MJCF importer; it is not a hand-authored replacement model.

The original MJCF, meshes, texture, and license are preserved under this
directory. Run `scripts/prepare_microwave_asset.sh` to regenerate the USD.
