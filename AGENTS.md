# AGENTS.md

This repository is valuable work in progress. Read this file before editing or
running the simulator.

## Repository and environment

- Repository: the directory containing this `AGENTS.md`
- Conda environment: developer-owned; activate it first or select it with
  `VLA_ISAACLAB_ENV`
- Isaac Lab: shared v2.0.2 checkout at `ISAACLAB_ROOT` (defaults to
  `/media/data-ssd/software/IsaacLab-v2.0.2`)
- Isaac Sim 4.5.0.0
- Python 3.10.21; PyTorch 2.5.1+cu121; LeRobot 0.4.3
- Tested GPU: NVIDIA RTX 4090, driver 550.144.03

Always begin simulator work with:

```bash
cd <vla_isaaclab-checkout>
conda activate <developer-env>
source scripts/activate.sh
check_install_environment
```

The activation script selects the configured local environment, disables
user-site packages, verifies the pinned shared Isaac Lab checkout, accepts the
Omniverse EULA, and places Kit caches inside the project. Each developer owns
their Conda environment and outputs; nobody clones or modifies Isaac Lab from
this project. Do not substitute a bare `conda activate`.

Before installing anything, inspect `which python`, `which pip`, and
`conda info --envs`. Do not use sudo, alter drivers/system CUDA, or touch other
users' files.

## Safety and working tree

- Preserve unrelated uncommitted work and generated outputs.
- Do not reset, clean, checkout, stash, or commit unless explicitly asked.
- Never fake a physical grasp or label a failed manipulation as successful.
- Do not teleport, parent, kinematically move, or invisibly attach an object.
- Large downloads require a disk-space check and an explanation first.

## Architecture

```text
Scenario
├── World       geometry and semantic frames
├── Robot       articulation, joints, limits, end effectors, reset pose
├── Objects     physical assets and semantic roles
├── Sensors     camera/contact definitions
├── Task        observations, reward, success, termination, task goal
├── Expert      optional scripted strategy and EE/gripper targets
├── Controller  reusable control and normalized action generation
└── Recorder    physical state, normalized action, sensors, metadata
```

Scripted runtime path:

```text
Task -> Expert -> EndEffectorTarget -> Controller -> normalized action
     -> env.step() -> Isaac Lab ActionTerm -> robot
```

There is no Command, Policy, or Action Adapter architecture layer. Shared action
mapping belongs in `actions.py` and the Isaac Lab ActionTerm.

Responsibility boundaries are strict:

- Tasks contain no phases, IK, trajectory generation, or joint control.
- Experts contain phases and grasp strategy but no Jacobians, differential IK,
  or action normalization.
- Controllers contain control/IK, joint limiting, and normalization but no task
  reward, success, phases, or object-specific strategy.
- `actions.py` contains only generic ActionTerm and mapping utilities.

## Key files

- `src/vla_isaaclab/contracts.py`: immutable component interfaces and targets.
- `src/vla_isaaclab/registry.py`: component registries.
- `src/vla_isaaclab/scenario.py`: compatibility validation and composition.
- `src/vla_isaaclab/runtime.py`: common environment runtime.
- `src/vla_isaaclab/actions.py`: normalized 25-joint ActionTerm and conversion.
- `src/vla_isaaclab/robots/unitree_g1.py`: fixed-base G1 definition.
- `src/vla_isaaclab/tasks/`: observations/reward/success/termination only.
- `src/vla_isaaclab/experts/`: scripted task-solving strategy only.
- `src/vla_isaaclab/controllers/`: reusable action generation.
- `src/vla_isaaclab/recording/`: HDF5 and LeRobot v3 pipelines.
- `scripts/run_scenario.py`: composition runner, not task/control logic.

## Reference compositions

Scene preview, with no Expert:

```text
World-Tabletop-v0
Robot-UnitreeG1-v0
Objects-YCB-Basic-v0
Sensors-FixedRGBD-v0
Task-ScenePreview-v0
Controller-Standing-v0
```

Sugar-box manipulation (the table contains only this object):

```text
World-Tabletop-v0
Robot-UnitreeG1-v0
Objects-YCB-SugarBox-v0
Sensors-FixedRGBD-v0
Task-YCBPickPlaceSugarBox-v0
Expert-YCBPickPlaceSugarBox-v0
Controller-LeftArmDifferentialIK-v0
```

Robot right is base `-Y`, equal to world `+X` for the fixed placement. The task
target is 0.02 m in world `-X` (robot left) from the deterministic initial sugar-box pose.
The sugar box is upright with world-Z yaw 35.81856 degrees, starting at
`(0.010509, -0.290489)` m. The hand uses a geometry-fitted side grasp with
world yaw about 65 degrees and 12 degrees downward pitch. The calibrated palm
offset from the box root is `(-0.107797, -0.110146, 0.075060)` m. All three
fingers receive closing motion; the thumb base retains its opposing pose.

The phase sequence is:

```text
reset -> move_to_turn_point -> orient_hand -> move_pregrasp -> close_gripper -> lift
      -> move_left -> level_box -> lower -> open_gripper -> retreat -> done
```

## Current validated state (2026-09-19)

- Package/imports, registries, and both reference compositions load.
- ScenePreview rendered 240 steps with stable G1/YCB state and working camera.
- HDF5 and LeRobot ScenePreview smoke episodes were inspected and replayed.
- The tabletop contains only `004_sugar_box` for this composition. The fixed G1
  base is at `(0.0, -0.64, 0.74)`. The torso starts centered; the left arm is
  initialized behind the box, above table height, with palm at approximately
  `(-0.1765, -0.4943, 0.7780)` m and the left hand open.
- The Expert first moves to turn point `(-0.200, -0.448594, 0.852068)` m,
  rotates there, then approaches. Closure requires <12 mm / <5 degrees for
  15 consecutive steps and holds the measured palm pose. Closure takes 120
  control steps; commanded lift is 8 cm. Torso yaw remains +/-25 degrees.
- Latest recording run reached Task success at step 961 after opening and
  withdrawal. The success XY tolerance is now 15 mm (was 8 mm), explicitly
  approved by the user. Height <15 mm, linear speed <0.04 m/s, angular speed
  <0.30 rad/s and 15 consecutive samples remain required. Palm distance must
  now exceed 20 cm, rather than 14 cm, to avoid accepting a held box.
- Release requires 5 consecutive samples with height error <8 mm, XY error
  <15 mm, and tilt <8 degrees; no pre-release velocity gate. Opening takes
  90 steps, followed by a hold until step 121 and horizontal withdrawal.
- Verified LeRobot v3 dataset contains 961 frames at 30 Hz, one episode,
  with next.success=true and next.done=true on the last frame.
  Dataset: `outputs/lerobot/sugar_box_left_place_20260919/`.
  Verified Parquet state/action finiteness, 961 decoded RGB frames, and an
  official LeRobotDataset API read. Validation passed=true, success_count=1.
  Recording log: `outputs/videos/sugar_box_lerobot_20260919/run.log`.
  Release preview: `outputs/videos/sugar_box_release_20260919/`.
  Offline USD geometry calculations: `outputs/analysis/sugar_box_three_finger/`.

Do not claim that the sugar-box task succeeds unless the Task's `success`
termination term fires, and never save a failed episode as a successful demo.

## Useful commands

```bash
./scripts/run_scenario.sh --headless --list-components
./scripts/run_scenario.sh --headless --objects Objects-YCB-Basic-v0 \
  --task Task-ScenePreview-v0 --controller Controller-Standing-v0 --steps 240
```

When adding functionality: add a Task for a new problem, add an Expert only for
a scripted solution, reuse an existing Controller when its motion interface
fits, and add a new Controller only for a genuinely new control modality.
