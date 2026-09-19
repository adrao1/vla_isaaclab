# vla_isaaclab

`vla_isaaclab` is a robotics simulation framework built on **Isaac Lab**, with
**Isaac Sim** serving as the underlying simulation runtime.

It provides:

- a reproducible Isaac Lab / Isaac Sim runtime
- composable robotic manipulation scenarios
- robot and asset definitions
- task and control interfaces
- scripted experts
- LeRobot v3 recording

The project separates scene composition, robot definitions, task logic, expert
strategies, low-level control, and dataset recording so that different
manipulation tasks can reuse the same simulation and control infrastructure.

## Architecture

A complete simulation `Scenario` is composed of the following components:

```text
Scenario
├── World       geometry and semantic frames
├── Robot       asset, joint groups, limits, end effectors, reset pose
├── Objects     physical assets and semantic roles
├── Sensors     camera and contact definitions
├── Task        observation, reward, success, termination, goal
├── Expert      optional scripted strategy and EE/gripper targets
├── Controller  reusable IK/control and normalized action generation
└── Recorder    state, action, image, and metadata export
```

For a scripted task:

```text
Task state -> Expert -> EndEffectorTarget -> Controller -> normalized action
           -> env.step() -> Isaac Lab ActionTerm -> robot
```

Component responsibilities are kept separate:

- `Task` defines task state, goals, rewards, success conditions, and
  termination, but does not contain task phases, IK, or joint-level control.
- `Expert` defines the high-level scripted strategy and produces end-effector
  and gripper targets, but does not implement Jacobians or action normalization.
- `Controller` implements reusable IK, robot control, and action mapping without
  containing object-specific or task-specific strategy.
- `Robot` defines the shared joint order, joint limits, joint groups,
  end-effector frames, and reset pose.
- `Recorder` independently handles state, action, image, and metadata recording
  and dataset export.

This separation allows Tasks, Experts, Robots, Objects, and Controllers to be
composed and reused without coupling task semantics to low-level robot control.

## Environment model

Isaac Lab is a shared external dependency on the lab server. All developers use
the same pinned Isaac Lab `v2.0.2` checkout:

```text
/media/data-ssd/software/IsaacLab-v2.0.2
```

Each developer maintains their own:

- Conda environment
- `vla_isaaclab` checkout
- `outputs/`
- Isaac Sim / Kit caches

| Component | Version |
| --- | --- |
| Python | 3.10 |
| Isaac Sim | 4.5.0.0 |
| Isaac Lab | v2.0.2 |
| PyTorch | 2.5.1+cu121 |
| LeRobot | 0.4.3 / Dataset v3 |

`environment.yml` defines the Conda base environment. `requirements.txt` pins
Isaac Sim, PyTorch, Isaac Lab runtime dependencies, and recording packages.

`ISAACLAB_ROOT` identifies the shared Isaac Lab source. It defaults to the
server path above in `scripts/activate.sh` and may be overridden on another
server. The shared checkout must exist and correspond to tag `v2.0.2`.
Developers do not clone or modify Isaac Lab themselves.

## Installation

Prerequisites:

- access to the shared Isaac Lab path
- Linux
- NVIDIA GPU and a driver compatible with Isaac Sim 4.5
- Git
- an initialized Miniconda or Mambaforge shell

The first dependency installation may download large NVIDIA Python packages,
but it does not download Isaac Lab.

### New developer setup

Choose your own Conda environment name:

```bash
git clone <this-repository-url> vla_isaaclab
cd vla_isaaclab

conda env create --name <your-env> --file environment.yml
conda activate <your-env>

export ISAACLAB_ROOT=/media/data-ssd/software/IsaacLab-v2.0.2
git -C "$ISAACLAB_ROOT" describe --tags --exact-match

python -m pip install -r requirements.txt
python -m pip install --no-deps lerobot==0.4.3
```

The Git command must print `v2.0.2`. Link the five shared Isaac Lab package
roots into the active developer-owned environment without writing into the
shared checkout:

```bash
python - <<'PY'
import os
import site
from pathlib import Path

root = Path(os.environ["ISAACLAB_ROOT"]).resolve()
packages = (
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_mimic",
    "isaaclab_rl",
    "isaaclab_tasks",
)
paths = [root / "source" / package for package in packages]
missing = [path for path in paths if not (path / path.name).is_dir()]
if missing:
    raise SystemExit(f"Missing shared Isaac Lab packages: {missing}")

link = Path(site.getsitepackages()[0]) / "isaaclab_shared.pth"
link.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
print(f"Wrote {link}")
PY

python -m pip install -e .
source scripts/activate.sh
check_install_environment
```

To update an existing environment after dependency changes:

```bash
conda activate <your-env>
conda env update --name <your-env> --file environment.yml
python -m pip install -r requirements.txt
python -m pip install --no-deps lerobot==0.4.3
python -m pip install -e .
```

Repeat the `.pth` linking step if `ISAACLAB_ROOT` changes or the environment is
recreated.

## Daily use

Activate your own environment first. `activate.sh` never guesses an environment
name and never falls back to another developer's environment:

```bash
conda activate <your-env>
source scripts/activate.sh
check_install_environment
./scripts/run_scenario.sh --headless --list-components
```

The run, record, replay, and asset-preparation wrappers source `activate.sh`
internally and preserve the already active non-`base` Conda environment. Once
the environment is active, the runner can also be used directly:

```bash
python scripts/run_scenario.py --headless --list-components
```

To select an environment explicitly instead:

```bash
export VLA_ISAACLAB_ENV=<your-env>
source scripts/activate.sh
```

## Project layout

```text
vla_isaaclab/
├── environment.yml              reproducible Conda base
├── requirements.txt             pinned Python/runtime packages
├── pyproject.toml               installable `vla_isaaclab` package
├── assets/                      USD, YCB, robot, and scene assets
├── configs/                     Kit and recording configuration
├── scripts/                     run, record, replay, asset utilities
├── src/vla_isaaclab/
│   ├── worlds/                  tables, pedestals, semantic workspace frames
│   ├── robots/                  robot definitions, joints, limits, EE frames
│   ├── objects/                 YCB and other object-set adapters
│   ├── sensors/                 RGB-D/contact configuration
│   ├── tasks/                   state, reward, success, termination
│   ├── experts/                 scripted task strategies
│   ├── controllers/             IK and normalized action generation
│   ├── recording/               HDF5 staging and LeRobot v3 export
│   ├── actions.py               normalized ActionTerm mapping
│   └── scenario.py              component compatibility and composition
├── tests/                       non-simulator unit tests
└── outputs/                     local caches, previews, recordings (ignored)
```

## Core interface conventions

The project maintains shared interface conventions so that different scenarios
and tasks can be composed consistently.

### Robot

`RobotDefinition` defines:

- joint names and joint order
- joint limits
- joint groups
- end-effector frame names
- reset pose
- action mapping

Tasks and controllers should not maintain independent robot joint definitions.

### Objects

Object definitions provide:

- asset source
- physical dimensions
- mass
- collision properties
- semantic role
- initial pose

Tasks should access object state through a consistent object interface rather
than depending on asset-specific USD internals.

### Coordinate frames

World, Robot, Object, and End Effector use explicit coordinate-frame
conventions. Scenario-specific semantic frames should be defined by World,
Robot, or Object components rather than being scattered across Task or Expert
implementations.

### Actions

Controllers convert:

```text
EndEffectorTarget
        ↓
physical robot target
        ↓
normalized simulation action
```

into actions executable by the Isaac Lab `ActionTerm`. Physical state and
normalized simulation actions are stored separately in recorded datasets.

## Component registration and selection

Each World, Robot, Objects, Sensors, Task, Expert, and Controller definition
has a unique `component_id`. The corresponding package `__init__.py` registers
the definition with `src/vla_isaaclab/registry.py`. At startup,
`register_defaults()` loads those registrations, and command-line values such
as `--world World-Tabletop-v0` and `--objects Objects-YCB-Basic-v0` select them
by ID. `src/vla_isaaclab/scenario.py` then checks component capabilities and
composes the selected modules into one scenario.

Use the following command to see every currently registered ID:

```bash
./scripts/run_scenario.sh --headless --list-components
```

When adding a component, define its `component_id`, import it in its module
package `__init__.py`, and add it to that package's `register_*()` function.

## Scene preview examples

The same preview command is used for all three scenes:

```bash
./scripts/run_scenario.sh --headless --enable_cameras \
  --world <WORLD_ID> \
  --objects <OBJECTS_ID> \
  --task Task-ScenePreview-v0 \
  --controller Controller-Standing-v0 \
  --steps 240 \
  --preview-video outputs/previews/<NAME>.mp4
```

Choose one option:

| Preview | `WORLD_ID` | `OBJECTS_ID` | `NAME` |
| --- | --- | --- | --- |
| YCB objects | `World-Tabletop-v0` | `Objects-YCB-Basic-v0` | `ycb` |
| Bowl and plate | `World-Tabletop-v0` | `Objects-Dinnerware-v0` | `dinnerware` |
| Microwave | `World-MicrowaveTabletop-v0` | `Objects-Microwave-v0` | `microwave` |

YCB and dinnerware share the tabletop layout and camera pose; only the Objects
component changes. The microwave selects a different World because it needs a
different robot distance, object placement, and camera pose.

## Reference scenario: YCB sugar box

The current deterministic reference task uses only:

```text
004_sugar_box
```

Implementation paths:

- Object registration: `src/vla_isaaclab/objects/ycb.py`
- Task and success conditions: `src/vla_isaaclab/tasks/ycb_pick_place_sugar_box.py`
- Scripted grasp strategy: `src/vla_isaaclab/experts/ycb_pick_place_sugar_box.py`
- Differential IK controller: `src/vla_isaaclab/controllers/left_arm_differential_ik.py`
- G1 robot/joint definition: `src/vla_isaaclab/robots/unitree_g1.py`
- Normalized joint ActionTerm: `src/vla_isaaclab/actions.py`
- Scenario runner and recording gate: `scripts/run_scenario.py`

The box starts:

- upright
- at XY position `(0.010509, -0.290489)` m
- with world-Z yaw `35.81856°`

The left hand performs a calibrated three-finger side grasp. It then:

1. grasps the sugar box
2. lifts it by 8 cm
3. moves it 2 cm toward the robot's left side, corresponding to world `-X`
4. levels the object
5. lowers it
6. releases it
7. withdraws the hand

Run:

```bash
./scripts/run_scenario.sh --headless --enable_cameras \
  --objects Objects-YCB-SugarBox-v0 \
  --task Task-YCBPickPlaceSugarBox-v0 \
  --expert Expert-YCBPickPlaceSugarBox-v0 \
  --controller Controller-LeftArmDifferentialIK-v0 \
  --steps 1800 \
  --preview-video outputs/videos/sugar_box_pick_place/attempt.mp4
```

The recorded reference run reached Task success at `step 961`. Success requires:

- XY error <= 15 mm
- height error <= 15 mm
- low object speed
- palm separation > 20 cm
- a 15-step hold after release

Failed sugar-box episodes are rejected by the LeRobot exporter and are not
included in the final training dataset.

## Recording

HDF5 is used as a debug/raw format. LeRobot v3 is the training format.

A LeRobot recording first writes an atomic local HDF5 staging file and then
automatically converts it to:

```text
outputs/lerobot/<dataset-name>/
```

No manual conversion step is required.

Example:

```bash
./scripts/run_scenario.sh --headless \
  --objects Objects-YCB-SugarBox-v0 \
  --task Task-YCBPickPlaceSugarBox-v0 \
  --expert Expert-YCBPickPlaceSugarBox-v0 \
  --controller Controller-LeftArmDifferentialIK-v0 \
  --record-format lerobot --episodes 1 --steps 1500 \
  --dataset-name sugar_box_demo
```

Replay a generated dataset with:

```bash
./scripts/replay_lerobot.sh outputs/lerobot/sugar_box_demo --episode 0 --headless
```

`outputs/` is intentionally ignored by Git. Validated datasets should be stored
in a dataset registry or object store rather than committed to the source
repository.
