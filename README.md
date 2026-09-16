# YCB tabletop simulation

This project provides a small Isaac Sim/Isaac Lab tabletop scene for testing perception and future pick-and-place work. It uses an isolated Conda environment and does not depend on another user's simulator installation.

## Installed versions

- Conda environment: `jiajunl_isaac`
- Environment path: `/home/vlakbnn/miniconda3/envs/jiajunl_isaac`
- Python: 3.10.21
- Isaac Sim: 4.5.0.0
- Isaac Lab source release: v2.0.2 (`isaaclab` package metadata reports 0.34.9 for this release)
- PyTorch: 2.5.1+cu121
- Tested GPU: NVIDIA GeForce RTX 4090, driver 550.144.03

## Activate the environment

From any Bash shell:

```bash
cd /home/vlakbnn/jiajunl4/YCB_Object
source scripts/activate.sh
```

The activation script selects the named environment from Conda's default environment directory, disables user-site Python packages, and keeps temporary files, Kit runtime data and shader caches under `outputs/runtime` inside this project. The script also defines `check_install_environment`, which prints and verifies `python`, `pip` and `conda info --envs` before an installation command.

## Run

Run a finite headless simulation and save validation results:

```bash
./scripts/validate_scene.sh
```

Run the separate bowl-and-plate scene and save its camera image:

```bash
./scripts/validate_dinnerware_scene.sh
```

Run the scene with a GUI for 600 physics steps:

```bash
./scripts/run_scene.sh --enable_cameras --steps 600
```

Keep the GUI open until its window is closed:

```bash
./scripts/run_scene.sh --enable_cameras --steps 0
```

On a server without an X display, use `--headless`. The first rendered run can take about one minute while Kit builds its local shader cache; later runs reuse it.

## Scene contents

- A 1.2 m by 0.8 m white table with a 0.75 m work-surface height and four collision-enabled legs
- Ground plane, dome light and directional light
- A 640 by 480 RGB/depth observation camera looking toward the work surface
- `/World/RobotSpawn` at `[0.0, -0.72, 0.0]`; its +Y direction faces the table
- Four rigid YCB objects:
  - `003_cracker_box`
  - `004_sugar_box`
  - `006_mustard_bottle`
  - `005_tomato_soup_can`

A second scene in `src/run_dinnerware_scene.py` places an official Isaac Sim bowl and large plate on the same white table. It uses the corresponding launchers in `scripts/run_dinnerware_scene.sh` and `scripts/validate_dinnerware_scene.sh`.

No robot model is selected because the final real robot has not been specified. To test a selected model without changing the scene code:

```bash
./scripts/run_scene.sh --enable_cameras --robot-usd /absolute/path/to/robot.usd
```

## Assets and physics

The objects come from NVIDIA's official Isaac Sim 4.5 asset server under:

`Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/`

Only the four selected assets and their referenced visual textures are cached under `assets/YCB`. Their source URLs, sizes and SHA-256 hashes are recorded in `assets/YCB/manifest.json`.

All four selected physics USD files contain a rigid body, collision geometry and mass properties. Their geometry references the corresponding official `Axis_Aligned` visual USD. The asset inspection result is saved to `outputs/asset-inspection.json`.

The originally considered pudding box and banana were replaced because Isaac Sim 4.5's official `Axis_Aligned_Physics` directory did not provide those two files. No substitute collision model was hand-authored.

The dinnerware scene uses NVIDIA Isaac Sim 4.5 ArchVis assets from:

`NVIDIA/Assets/ArchVis/Residential/Kitchen/Kitchenware/Dinnerware/`

The selected files are `bowl_plate.usd` and `plate_large.usd`. Their USD files, material textures and hashes are cached under `assets/Isaac` and recorded in `assets/Isaac/manifest.json`. These visual assets do not contain physics schemas, so the scene supplies explicit rigid bodies, masses and hidden solid-cylinder collision proxies. The proxies model stable outer support shapes; they do not model the bowl's concave interior.

## Validation outputs

- `outputs/scene_rgb.png`: RGB camera observation
- `outputs/ycb_pick_place_scene.usd`: composed scene before simulation starts
- `outputs/physics-validation.json`: final object poses, velocities and pass/fail checks
- `outputs/asset-inspection.json`: static USD rigid-body, collision, bounds and mass inspection
- `outputs/validate-scene.log`: full launch and validation log
- `outputs/dinnerware_scene_rgb.png`: bowl-and-plate RGB observation
- `outputs/dinnerware_scene.usd`: composed bowl-and-plate scene
- `outputs/dinnerware-physics-validation.json`: dinnerware physics validation
- `outputs/dinnerware-asset-inspection.json`: original asset units, bounds and schema inspection

The validation runs 1200 steps at 120 Hz. It fails if an object has a non-finite state, falls through the table, retains a combined linear/angular speed of at least 0.08 after settling, or has a non-positive runtime mass/inertia. PhysX computes the runtime inertia from each official collision shape and authored mass.

## Project layout

```text
YCB_Object/
├── README.md
├── IsaacLab/          # pinned v2.0.2 source checkout
├── assets/YCB/        # selected official YCB assets and manifest
├── assets/Isaac/      # selected official dinnerware assets and manifest
├── configs/           # scene data and minimal Kit experiences
├── outputs/           # reports, image, USD and logs
├── scripts/           # activation, launch, inspection and validation
└── src/               # scene and asset-cache implementation
```

## Known issues

- The current scene exposes a robot spawn interface but intentionally does not choose a robot model.
- On a headless server, Isaac Sim logs GLFW/display warnings even though Vulkan rendering and physics run successfully.
- The mustard bottle can settle on its side after being dropped. This is physically valid and useful for later pose-randomization tests.
- Hosted NVIDIA assets require network access only when a selected file is not already cached locally.

## Next steps

1. Select the real robot and pass its USD through `--robot-usd`; then add its joint and gripper configuration in a separate robot configuration module.
2. Add a controller interface for joint-space or Cartesian end-effector commands.
3. Use the existing RGB/depth camera to expose observations and calibration data.
4. Randomize initial object poses within the table bounds and record ground-truth poses.
5. Add grasp targets and a scripted pick-and-place baseline before introducing RL or imitation learning.
6. Add domain randomization only after the deterministic controller and perception pipeline pass repeatable tests.
