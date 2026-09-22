# vla_isaaclab

`vla_isaaclab` provides registered Isaac Lab environments for VLA simulation,
scripted data generation, and future real-to-sim / sim-to-real projects. Isaac
Sim is the runtime; Isaac Lab supplies the environment and manager framework.

## Environment model

Isaac Lab is a shared, read-only external dependency on the lab server:

```text
shared
└── /media/data-ssd/software/IsaacLab-v2.0.2

per developer
├── own Conda environment
├── own vla_isaaclab checkout
├── own real2sim / sim2real checkouts
└── own outputs and Kit caches
```

All developers use the pinned Isaac Lab `v2.0.2` checkout. They do not clone or
modify Isaac Lab inside this repository. Each adjacent project is installed in
the developer's own Conda environment and can import or launch these registered
environments without sharing project source trees or outputs.

| Component | Version |
| --- | --- |
| Python | 3.10 |
| Isaac Sim | 4.5.0.0 |
| Isaac Lab | v2.0.2 |
| PyTorch | 2.5.1+cu121 |
| LeRobot | 0.4.3 / Dataset v3 |

## Installation

Prerequisites are Linux, an NVIDIA GPU/driver compatible with Isaac Sim 4.5,
Git, Conda, and read access to the shared Isaac Lab checkout.

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

The Git command must print `v2.0.2`. Link the shared packages into this Conda
environment without writing to the shared checkout:

```bash
python - <<'PY'
import os
import site
from pathlib import Path

root = Path(os.environ["ISAACLAB_ROOT"]).resolve()
packages = ("isaaclab", "isaaclab_assets", "isaaclab_mimic", "isaaclab_rl", "isaaclab_tasks")
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

There is intentionally no bootstrap script. `activate.sh` never guesses a
developer environment: activate it first, or explicitly set
`VLA_ISAACLAB_ENV=<your-env>`.

## Daily use

```bash
conda activate <your-env>
source scripts/activate.sh
check_install_environment
./scripts/run_env.sh --headless --physics-only --list-tasks
```

Wrappers source `activate.sh` but preserve the already selected non-`base`
environment. `ISAACLAB_ROOT` defaults to the shared server path and remains
overridable.

## Architecture

The selectable unit is one complete Gym environment, not independently
registered World/Object/Task/Controller components:

```text
Gym environment ID
└── concrete EnvCfg
    ├── InteractiveSceneCfg     G1, table, object(s), camera
    ├── ActionManager          normalized 25-D joint action
    ├── CommandManager         task goal, when applicable
    ├── ObservationManager     policy/task observations
    ├── EventManager           reset and uncontrolled-joint targets
    ├── RewardManager          task rewards
    ├── TerminationManager     success/failure/time-out
    └── RecorderManager        raw HDF5 state/action/camera recording
```

The scripted sugar-box path is:

```text
registered EnvCfg + manager state
        ↓
scripted policy (task phases + bounded IK + 3-finger targets)
        ↓
physical joint targets
        ↓  q -> 2 * (q - lower) / (upper - lower) - 1
normalized 25-D action
        ↓
Isaac Lab JointPositionToLimitsActionCfg
        ↓
G1 joint position targets
```

The environment interface is normalized joint action, not Cartesian pose. The
custom ActionTerm and Controller registry were removed. Isaac Lab's
`JointPositionToLimitsActionCfg` performs the action mapping directly. The only
custom control algorithm retained is the bounded damped-least-squares IK used
inside the scripted sugar-box policy.

## Project layout

```text
vla_isaaclab/
├── assets/                          local robot, YCB, dinnerware, microwave assets
├── configs/                         Isaac Sim Kit configurations
├── scripts/
│   ├── activate.sh                 environment verification and local caches
│   ├── run_env.py/.sh              generic Gym environment runner
│   ├── record_lerobot.sh            LeRobot v3 data generation
│   ├── replay_lerobot.py/.sh        normalized-action replay
│   └── prepare_*.py/.sh             asset preparation utilities
├── src/vla_isaaclab/
│   ├── envs/
│   │   ├── common/
│   │   │   ├── base.py             common simulation timing/material settings
│   │   │   ├── g1.py               reusable G1 config and joint semantics
│   │   │   ├── managers.py         built-in action, observations, reset events
│   │   │   └── scene.py            shared table/light/camera helpers
│   │   ├── scene_preview/
│   │   │   ├── __init__.py         three Gym registrations
│   │   │   └── env_cfg.py          YCB, dinnerware, microwave complete scenes
│   │   └── ycb_sugar_box/
│   │       ├── __init__.py         sugar-box Gym registration
│   │       ├── env_cfg.py          G1 + one sugar box + all manager configs
│   │       └── mdp/                 command, observation, reward, termination terms
│   ├── policies/
│   │   ├── standing.py             normalized default-pose policy
│   │   ├── ycb_sugar_box*.py       scripted phases and action generation
│   │   ├── bounded_ik.py           retained custom IK helper
│   │   └── joint_limits.py         inverse of the built-in action mapping
│   └── recording/                   RecorderManager HDF5 and LeRobot v3 pipeline
├── tests/                           simulator-free unit tests
└── outputs/                         local caches/videos/datasets; ignored by Git
```

Robot and reusable table/camera helpers live under `envs/common`. Object assets
and task-specific scene configuration live in the concrete environment config;
there are no separate top-level robot/object/world/sensor registries.

## Registration and extension

Importing `vla_isaaclab` registers these IDs with Gymnasium:

- `VLA-ScenePreview-YCB-G1-v0`
- `VLA-ScenePreview-Dinnerware-G1-v0`
- `VLA-ScenePreview-Microwave-G1-v0`
- `VLA-YCBSugarBox-G1-JointPos-v0`

To add a task, create a complete EnvCfg under `src/vla_isaaclab/envs/`, keep its
MDP terms beside it, and register the EnvCfg in that environment package's
`__init__.py`. Add a scripted policy only when deterministic demonstrations are
needed. Do not add another component registry or a custom ActionTerm for a
mapping already provided by Isaac Lab.

Camera placement belongs to the concrete scene's EnvCfg because framing depends
on the object layout. Shared camera intrinsics/modalities belong in the common
camera helper. Thus, change `CAMERA_EYE`/`CAMERA_TARGET` in the concrete env to
reframe one task, and change `camera_cfg()` to alter camera hardware across all
tasks.

## Scene preview examples

Use one command and choose one of three task IDs:

```bash
./scripts/run_env.sh --headless \
  --task <TASK_ID> --steps 240 \
  --preview-video outputs/previews/<NAME>.mp4
```

| Preview | `TASK_ID` | `NAME` |
| --- | --- | --- |
| YCB objects | `VLA-ScenePreview-YCB-G1-v0` | `ycb` |
| Bowl and plate | `VLA-ScenePreview-Dinnerware-G1-v0` | `dinnerware` |
| Microwave | `VLA-ScenePreview-Microwave-G1-v0` | `microwave` |

## Reference task: YCB sugar box

The table contains only `004_sugar_box`. Relevant code is:

- complete environment: `src/vla_isaaclab/envs/ycb_sugar_box/env_cfg.py`
- MDP terms: `src/vla_isaaclab/envs/ycb_sugar_box/mdp/`
- G1 configuration: `src/vla_isaaclab/envs/common/g1.py`
- built-in action configuration: `src/vla_isaaclab/envs/common/managers.py`
- scripted phases: `src/vla_isaaclab/policies/ycb_sugar_box_strategy.py`
- IK and action generation: `src/vla_isaaclab/policies/ycb_sugar_box.py`
- bounded IK helper: `src/vla_isaaclab/policies/bounded_ik.py`

The box starts upright at XY `(0.010509, -0.290489)` m with world-Z yaw
`35.81856°`. The three-finger side grasp lifts it 8 cm, moves it 2 cm toward
robot-left (world `-X`), levels and releases it, then withdraws the hand.

```bash
./scripts/run_env.sh --headless \
  --task VLA-YCBSugarBox-G1-JointPos-v0 \
  --steps 1200 \
  --preview-video outputs/videos/sugar_box/attempt.mp4
```

The validated reference fires the named `success` termination at step 961.
Success requires <=15 mm XY/height error, low object speed, palm separation
over 20 cm, and a 15-step hold. A failed run must never be labeled or saved as
a successful demonstration.

## Recording

Generate a LeRobot v3 dataset directly; no user-run conversion step is needed:

```bash
./scripts/record_lerobot.sh --dataset-name sugar_box_demo
```

The dataset contains physical joint state/targets, normalized simulator action,
environment state, phase, reward/done/success, seed, and RGB. The writer stages
an episode atomically, rejects unsuccessful sugar-box episodes, then converts
to `outputs/lerobot/<dataset-name>/`.

Replay normalized actions with:

```bash
./scripts/replay_lerobot.sh outputs/lerobot/sugar_box_demo --episode 0 --headless
```

`outputs/` is ignored by Git. Publish validated datasets to a dataset registry
or object store rather than committing them.
