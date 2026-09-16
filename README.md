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

Run the Unitree G1 bowl-and-plate scene and save its camera image:

```bash
./scripts/validate_dinnerware_scene.sh
```

Run the refactored manager-based task with a small scripted waist/arm motion:

```bash
./scripts/run_managed_task.sh --headless --steps 300 --demo-motion
```

Record a 120-control-step episode containing actions, states, observations, RGB, depth and calibration:

```bash
./scripts/record_managed_demo.sh --dataset-name g1_dinnerware_demo
python scripts/inspect_dataset.py outputs/datasets/g1_dinnerware_demo.hdf5
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

A second scene in `src/run_dinnerware_scene.py` places an official Unitree G1 in front of the same white table with an official Isaac Sim bowl and large plate. It uses the corresponding launchers in `scripts/run_dinnerware_scene.sh` and `scripts/validate_dinnerware_scene.sh`.

The G1 uses Isaac Lab v2.0.2's official `G1_CFG` and `Robots/Unitree/G1/g1.usd` asset. Its root is fixed at the standing pose and all joints receive their default position targets. The `torso_joint` (waist yaw), shoulder, elbow and hand/finger joints remain actuated so later control code can command them without changing the scene setup.

## Manager-based task architecture

The control/data path is implemented as an Isaac Lab `ManagerBasedRLEnv`. It is useful for scripted control, teleoperation and imitation-learning data even before an RL policy exists:

```text
controller or policy (30 Hz, 25 normalized upper-body actions)
    -> ActionManager (upper-body targets + lower-body standing hold)
    -> PhysX simulation (120 Hz, decimation 4)
    -> ObservationManager (89 floating-point values)
    -> Termination/Reward managers (task placeholders)
    -> RecorderManager (HDF5 actions, state, RGB/depth and task metadata)
```

Code ownership is split by responsibility:

- `src/ycb_sim/env_cfg.py`: scene, rates, observations, reward and termination configuration
- `src/ycb_sim/actions.py`: 25-dimensional G1 waist/arms/hands action mapping and lower-body hold
- `src/ycb_sim/controllers.py`: controller interface and safe scripted motion example
- `src/ycb_sim/mdp.py`: task observations, distance reward and success condition
- `src/ycb_sim/recording.py`: Isaac Lab recorder terms for camera calibration, frames and metrics
- `src/ycb_sim/spawners.py`: official dinnerware visuals and explicit physics proxies
- `src/run_managed_task.py`: application launch, control loop, screenshot and report

The current placeholder task succeeds when the bowl center is within 8 cm horizontally of the plate center. It does not yet implement grasping. Replace `StandingUpperBodyController.compute()` with a Cartesian controller, teleoperation source or policy adapter while keeping the scene and recorder unchanged.

The original YCB-only entry point remains model-neutral because the final real robot has not been specified. To test another selected model without changing that scene code:

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
- `outputs/dinnerware_g1_scene_rgb.png`: G1 with bowl-and-plate RGB observation
- `outputs/dinnerware_g1_scene.usd`: composed G1 dinnerware scene
- `outputs/dinnerware-g1-validation.json`: G1 joint list and dinnerware physics validation
- `outputs/managed_g1_dinnerware_rgb.png`: manager-based scene camera image
- `outputs/managed_g1_dinnerware_scene.usd`: manager-based composed scene
- `outputs/managed-task-validation.json`: rates, action mapping, lower-body hold and task state
- `outputs/datasets/*.hdf5`: recorded episodes (ignored by Git because they grow quickly)

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
└── src/
    ├── ycb_sim/       # manager-based scene, control, task and recorder modules
    └── *.py           # launchers and legacy validated scene entry points
```

## Known issues

- The YCB scene still exposes a model-neutral robot spawn interface; the dinnerware task explicitly uses Unitree G1 as requested.
- On a headless server, Isaac Sim logs GLFW/display warnings even though Vulkan rendering and physics run successfully.
- The mustard bottle can settle on its side after being dropped. This is physically valid and useful for later pose-randomization tests.
- Hosted NVIDIA assets require network access only when a selected file is not already cached locally.
- The G1 base is fixed and the lower body is held at its default pose. This is appropriate for tabletop controller development but does not model balance.
- The solid-cylinder bowl proxy models stable support, not the concave interior. Placing another object inside the bowl needs a convex-decomposition collider.
- Camera frames are stored without HDF5 compression. At 640x480 RGB plus float32 depth, recordings use about 2.1 MB per control step; check disk space before long capture sessions.

## Next steps

1. Add left/right end-effector frames and a differential inverse-kinematics controller behind the existing controller interface.
2. Add a gripper abstraction for the G1 hand and define pre-grasp, grasp, lift and place phases.
3. Add reset-time object pose randomization with deterministic seeds and workspace bounds.
4. Implement a scripted pick-and-place baseline and use the existing recorder for successful demonstrations.
5. Add dataset replay and quality checks before training an imitation-learning policy.
6. Replace the G1 configuration if the final physical robot differs, then validate joint ordering, limits and camera extrinsics.
