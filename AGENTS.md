# AGENTS.md

This file is the handoff guide for agents working in this repository. Treat the
current working tree as valuable work in progress. Read this file before editing
or running the simulator.

## Repository and environment

- Repository: `/home/vlakbnn/jiajunl4/isaaclab_manipulation`
- Required Conda environment: `jiajunl_isaac`
- Environment prefix: `/home/vlakbnn/miniconda3/envs/jiajunl_isaac`
- Python: 3.10.21
- Isaac Sim: 4.5.0.0
- Isaac Lab: v2.0.2, editable checkout at `./IsaacLab`
- PyTorch: 2.5.1+cu121
- LeRobot: 0.4.3, Dataset v3.0
- Tested hardware: NVIDIA RTX 4090, driver 550.144.03

Always begin with:

```bash
cd /home/vlakbnn/jiajunl4/isaaclab_manipulation
source scripts/activate.sh
check_install_environment
```

`scripts/activate.sh` activates the exact environment prefix, restores its
`python` and `pip` ahead of `~/.local/bin`, disables user-site packages, clears
`PYTHONPATH`, accepts the Omniverse EULA, and redirects Kit caches and temporary
files into this project. Do not replace it with a bare `conda activate` when
running project commands.

Before every `pip` or `conda` install, run all three checks below and confirm
that both executables come from `jiajunl_isaac`:

```bash
which python
which pip
conda info --envs
```

Do not install into `base`, system Python, another user's environment, or any
other Isaac Sim/Isaac Lab installation. Do not use `sudo`, change the NVIDIA
driver or system CUDA, edit shell startup files, or touch other users' files.
Large downloads require a disk-space check and an explanation to the user first.

## Git and file safety

- The last committed baseline is `a60ca20 Use YCB dinnerware assets and split baseline controllers`.
- The working tree contains substantial uncommitted bowl-task, controller,
  recording, action, robot-pose, world, and runner changes.
- Do not reset, checkout, clean, stash, overwrite, or discard these changes.
- Do not commit unless the user explicitly asks for a commit.
- Generated outputs and downloaded/generated assets are mostly ignored. Do not
  delete outputs unless the user explicitly asks which files may be removed.
- Never represent a failed manipulation run as a successful demonstration.

Inspect the state before work:

```bash
git status --short
git diff --stat
```

## Project purpose

This is a composable Isaac Lab manipulation platform. The current development
target is a fixed-base Unitree G1 using only its left arm and left hand to pick
up a physical YCB bowl, move it to a YCB plate, release it, and retreat. Grasping
must come from simulated contacts and friction. Never teleport the bowl, make it
kinematic, parent it to the hand, or attach it with a hidden/fixed joint.

The longer-term architecture must support multiple worlds, robots, object sets,
sensor rigs, tasks, controllers, policies, and recorders without putting all
logic in the runner.

## Architecture

`scripts/run_scenario.py` is the only main scenario entry point. Shell wrappers
activate and validate the environment before calling Python.

```text
ScenarioSelection
  -> registry capability validation
  -> scenario composer
  -> ManagerBasedRLEnvCfg / ScenarioEnv
     ├── World       geometry, support surfaces, semantic locations
     ├── Robot       USD, joint groups, end-effectors, reset pose
     ├── Objects     physical assets and semantic roles
     ├── Sensors     camera/contact sensor definitions
     ├── Task        observations, rewards, success and termination
     ├── Controller  scripted/IK/policy action source
     └── Recorder    stable frame schema, staging, LeRobot/HDF5 output
```

Key files:

- `src/sim_platform/contracts.py`: immutable component interfaces.
- `src/sim_platform/registry.py`: registries and component lookup.
- `src/sim_platform/scenario.py`: capability checks and environment composition.
- `src/sim_platform/runtime.py`: `ScenarioEnv` runtime wrapper.
- `src/sim_platform/actions.py`: 25-dimensional normalized upper-body action
  term and physical joint-target conversion.
- `src/sim_platform/worlds/`: reusable scene geometry and semantic positions.
- `src/sim_platform/robots/unitree_g1.py`: G1 asset, joint groups, fingertips,
  fixed base, solver settings, and raised left-arm reset pose.
- `src/sim_platform/objects/`: dinnerware, standard YCB set, and empty set.
- `src/sim_platform/sensors/fixed_rgbd.py`: fixed 640x480 RGB-D camera.
- `src/sim_platform/tasks/`: manager-based observations/rewards/terminations.
- `src/sim_platform/controllers/`: independent action sources and state machines.
- `src/sim_platform/recording/`: frame adapter, neutral HDF5 staging, LeRobot v3
  writer, and native HDF5 support.
- `scripts/run_scenario.py`: CLI, execution, preview video, reports, and dataset
  orchestration. Keep task and controller behavior out of this file.

Registered component IDs:

| Kind | IDs |
|---|---|
| Worlds | `World-Tabletop-v0`, `World-Pedestal-v0`, `World-MicrowaveTabletop-v0` |
| Robots | `Robot-UnitreeG1-v0` |
| Objects | `Objects-Dinnerware-v0`, `Objects-YCB-Basic-v0`, `Objects-Microwave-v0`, `Objects-None-v0` |
| Sensors | `Sensors-FixedRGBD-v0` |
| Tasks | `Task-PickPlace-v0`, `Task-Reach-v0`, `Task-BowlToPlate-v0`, `Task-ScenePreview-v0` |
| Controllers | `Controller-Standing-v0`, `Controller-RaiseLower-v0`, `Controller-LeftHandBowlToPlate-v0` |

Add a component in its own module, export/register it in that component package's
`__init__.py`, and declare capabilities in the contract. Tasks must obtain robot
joint and body names through `RobotDefinition`; do not hard-code G1 joint indices
inside a task.

## Current tabletop scene

- White table size: 1.2 x 0.8 x 0.05 m.
- Table/support height: 0.62 m.
- Robot root: `(0.0, -0.62, 0.74)`, fixed base, facing the table.
- Bowl spawn: `(-0.18, -0.28, 0.70)`; it settles with root Z near 0.62 m.
- Plate spawn: `(0.12, -0.24, 0.70)`; it settles with root Z near 0.62 m.
- Camera eye: `(0.0, 2.20, 2.25)`, target `(0.0, -0.25, 0.86)`.
- Physics: 120 Hz; control: 30 Hz; camera: 30 Hz.

The left arm starts raised and safely behind the bowl. Its reset joint positions
are defined by `LEFT_ARM_RAISED_JOINT_POS` in `robots/unitree_g1.py`. The latest
standing validation measured the palm at approximately
`(-0.1915, -0.5387, 0.7459) m`, about 7.3 cm above the published bowl rim, while
the bowl remained stable. The right arm and hand remain at their default standing
pose. The root and lower body remain fixed/held.

## Assets and physics

Dinnerware assets:

- `assets/YCB/dinnerware/024_bowl/024_bowl_physics.usd`
- `assets/YCB/dinnerware/029_plate/029_plate_physics.usd`
- Manifest: `assets/YCB/dinnerware/manifest.json`

These are official YCB 16k textured scans, normalized without scaling and
converted to USD. Published values are:

- Bowl: 0.159 x 0.159 x 0.053 m, 0.147 kg.
- Plate: 0.258 x 0.258 x 0.024 m, 0.279 kg.

Both use rigid-body gravity, authored mass, and convex-decomposition collision.
The object adapter enables contact sensors, collision, solver iterations, and
small contact/rest offsets. Do not replace these with hand-modeled primitives
unless the user requests it.

## Bowl-to-plate task

Task ID: `Task-BowlToPlate-v0`

Instruction:

```text
Use only the left hand to pick up the bowl and place it securely in the center of the plate.
```

The task is controller-independent. Its 83-dimensional policy observation
contains:

- left palm pose;
- seven left-hand joint positions;
- bowl pose, linear velocity, and angular velocity;
- plate pose;
- bowl pose relative to the hand and plate;
- ten-state controller phase one-hot vector;
- last 25-dimensional action.

It also creates filtered contact sensors for `left_two_link`, `left_four_link`,
and `left_six_link` against the bowl.

Strict success requires all of the following for 15 consecutive control steps:

- bowl/plate horizontal center distance below 4 cm;
- bowl root 0.8 to 7.5 cm above the plate root;
- bowl tilt below 15 degrees;
- bowl linear speed below 0.035 m/s;
- bowl angular speed below 0.25 rad/s;
- left hand more than 0.13 m from the bowl after release;
- bowl remains above the table.

## Bowl controller and current status

Controller ID: `Controller-LeftHandBowlToPlate-v0`

State machine:

```text
settle -> pre_grasp -> approach -> close_hand -> lift
       -> transfer -> lower -> release -> retreat -> done
```

The controller uses position-only damped least-squares differential IK on the
five left-arm joints, with a null-space reset-posture preference. Waist yaw is
reserved for transfer/lower/release. Joint commands are converted back through
the normalized action interface. Right arm/right hand commands stay at their
default targets. Hand commands use the seven physical G1 hand joints.

Current status as of 2026-09-18:

- Architecture, task observations/rewards/termination, phase state machine,
  contact sensors, diagnostics, preview-video support, and recording schema exist.
- Table-edge object placement and the raised initial arm pose have been physically
  validated.
- The full bowl-to-plate task has **not succeeded**.
- Latest forced debug trajectory completed `approach`, `close_hand`, and `lift`.
- `approach` timed out with about 0.074 m palm-position error.
- Finger-to-bowl contact force remained 0 N.
- The bowl stayed at Z 0.620 m and was not lifted.
- This indicates that the next work should focus on reachable approach geometry,
  palm orientation/fingertip placement, and IK branch/control, before tuning grip
  force or transfer.

Latest debug artifacts:

- `outputs/previews/bowl_grasp_lift_debug.mp4` (15 s, 450 frames, H.264)
- `outputs/previews/bowl_grasp_lift_approach.png`
- `outputs/previews/bowl_grasp_lift_close.png`
- `outputs/previews/bowl_grasp_lift_raise.png`
- Bowl-task validation report under the matching directory in `outputs/scenarios/`.

Diagnostic utilities:

- `scripts/inspect_g1_hand.py`: joint limits and link/body inspection.
- `scripts/probe_g1_jacobian.py`: finite-difference check showing that the G1
  palm translation Jacobian rows are reported in world axes in this setup.
- `scripts/search_g1_ik_seed.py`: parallel workspace/IK seed search, including
  optional `--center` and `--spread` joint-space sampling.

## Running and debugging

List registered components:

```bash
./scripts/run_scenario.sh --headless --list-components
```

Fast physics/controller run without cameras or RTX rendering:

```bash
./scripts/run_scenario.sh \
  --headless --physics-only --no-image \
  --task Task-BowlToPlate-v0 \
  --controller Controller-LeftHandBowlToPlate-v0 \
  --steps 450
```

Reproduce the latest approach/close/lift debug video:

```bash
./scripts/run_scenario.sh \
  --headless \
  --task Task-BowlToPlate-v0 \
  --controller Controller-LeftHandBowlToPlate-v0 \
  --steps 450 \
  --stop-after-phase lift \
  --debug-continue-on-timeout \
  --preview-video outputs/previews/bowl_grasp_lift_debug.mp4 \
  --no-image
```

`--debug-continue-on-timeout` exists only to visualize later phases after an
earlier failure. Do not use it to claim task success. `--preview-video` writes an
MP4 without creating a dataset.

Check the safe raised reset pose:

```bash
./scripts/run_scenario.sh \
  --headless \
  --task Task-PickPlace-v0 \
  --controller Controller-Standing-v0 \
  --steps 120
```

GUI runs require a display. A remote SSH session normally needs X forwarding,
VirtualGL, or another remote desktop solution. Headless camera rendering still
uses the GPU and does not need an interactive display.

## Reports and output semantics

Scenario runs write beneath:

```text
outputs/scenarios/<scenario-id>/
├── scene.usd
├── rgb.png
└── validation.json
```

Preview videos go under `outputs/previews/`. Isaac Sim runtime caches are under
`outputs/runtime/`. Do not interpret an old output as evidence for current code;
check timestamps and rerun the exact command.

The bowl controller's validation diagnostics include phase history, palm targets,
joint positions, contact-force maxima, debug warnings, final bowl/plate poses,
and failure reason. The runner also reports the left end-effector position and
height above the published object top.

## LeRobot recording pipeline

The pipeline is:

```text
Isaac Sim process
  -> ScenarioFrameAdapter
  -> atomic neutral HDF5 staging file
  -> separate conversion process
  -> LeRobot Dataset v3 directory
```

The process boundary avoids loading PyAV/TorchVision and Omniverse native
libraries together. The frame schema currently includes:

- 25 joint positions and velocities;
- 25 physical joint-position targets as `action`;
- 25 normalized simulator actions for exact replay;
- 40-value environment state: two 7D end-effector poses plus pose and 6D
  velocity for both rigid objects;
- RGB video;
- controller phase, reward, done, success, seed, and task prompt.

For `Task-BowlToPlate-v0`, `scripts/run_scenario.py` refuses to save an episode
unless the strict task termination reports success. Keep this guard. Record the
requested two successful episodes only after a real physical grasp and placement
works end to end.

After a successful controller exists, use a new dataset name and inspect it:

```bash
./scripts/run_scenario.sh \
  --headless \
  --task Task-BowlToPlate-v0 \
  --controller Controller-LeftHandBowlToPlate-v0 \
  --record-format lerobot \
  --episodes 2 \
  --steps 900 \
  --dataset-name g1_bowl_to_plate_success

python scripts/inspect_lerobot.py outputs/lerobot/g1_bowl_to_plate_success
```

Never weaken the success guard merely to produce a dataset.

## Validation expectations

Use the smallest useful check first:

1. Run `python -m py_compile` on changed Python files.
2. Use `--physics-only` for IK, contact, and state-machine iteration.
3. Run a headless rendered trajectory when visual evidence is needed.
4. Decode generated videos with `ffprobe` and extract representative frames.
5. Inspect `validation.json`, especially phase history, contacts, rigid-object
   stability, final poses, and success/timeout counts.
6. Run the LeRobot inspector only after a genuinely successful recording.

GPU runs may coexist with other users' jobs. Check `nvidia-smi`, do not kill or
modify unrelated processes, and prefer physics-only runs while tuning control.

## Known documentation drift

`README.md`, `commands.md`, and `configs/dinnerware_scene.json` still describe
parts of the earlier baseline and contain stale object positions, table height,
component lists, or recording dimensions. Treat executable source and this file
as the current state, then update those documents before final delivery.

## Immediate next steps

1. Use the latest debug video and phase report to choose a closer, reachable
   approach direction and palm orientation.
2. Add orientation control or a better IK seed/solver while keeping task logic
   independent of G1 joint indices.
3. Verify actual fingertip link geometry and open/closed directions at the bowl.
4. Obtain nonzero fingertip-to-bowl contact without disturbing the bowl.
5. Tune friction/contact/solver behavior only after geometry is correct.
6. Validate physical lift, transfer, lower, release, retreat, and strict success.
7. Record and inspect two successful LeRobot v3 episodes.
8. Update `README.md` and `commands.md` with the final commands and measured status.
