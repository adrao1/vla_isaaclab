# YCB dinnerware

This directory contains project-local conversions of two objects from the
[YCB Object and Model Set](https://ycb-benchmarks.s3.amazonaws.com/index.html):

- `024_bowl`: 16k textured scan, published mass 0.147 kg
- `029_plate`: 16k textured scan, published mass 0.279 kg

The YCB dataset is licensed under CC BY 4.0. `manifest.json` records the exact
download URLs, SHA-256 checksums, published dimensions, measured mesh bounds and
coordinate normalization. Run `scripts/prepare_ycb_dinnerware.sh` to reproduce
the local files. Isaac Lab's MeshConverter adds a rigid body, measured mass and
convex decomposition collision to each generated USD.
