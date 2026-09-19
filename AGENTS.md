# AGENTS.md

This repository is valuable work in progress. Read this file before editing or
running the simulator.

## Repository and environment

- Repository: `/home/vlakbnn/jiajunl4/isaac_simlab`
- Conda environment: `jiajunl_isaac`
- Prefix: `/home/vlakbnn/miniconda3/envs/jiajunl_isaac`
- Isaac Sim 4.5.0.0; Isaac Lab v2.0.2 in `./IsaacLab`
- Python 3.10.21; PyTorch 2.5.1+cu121; LeRobot 0.4.3
- Tested GPU: NVIDIA RTX 4090, driver 550.144.03

Always begin simulator work with:

```bash
cd /home/vlakbnn/jiajunl4/isaac_simlab
source scripts/activate.sh
check_install_environment
```

The activation script selects the exact environment, disables user-site
packages, clears `PYTHONPATH`, accepts the Omniverse EULA, and places Kit caches
inside the project. Do not substitute a bare `conda activate`.

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

- `src/isaac_simlab/contracts.py`: immutable component interfaces and targets.
- `src/isaac_simlab/registry.py`: component registries.
- `src/isaac_simlab/scenario.py`: compatibility validation and composition.
- `src/isaac_simlab/runtime.py`: common environment runtime.
- `src/isaac_simlab/actions.py`: normalized 25-joint ActionTerm and conversion.
- `src/isaac_simlab/robots/unitree_g1.py`: fixed-base G1 definition.
- `src/isaac_simlab/tasks/`: observations/reward/success/termination only.
- `src/isaac_simlab/experts/`: scripted task-solving strategy only.
- `src/isaac_simlab/controllers/`: reusable action generation.
- `src/isaac_simlab/recording/`: HDF5 and LeRobot v3 pipelines.
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
target is 0.02 m in world `+X` from the deterministic initial sugar-box pose.
The Expert must rise first, translate above the box at a safe height, point the
fingers downward, and only then descend to grasp.

## Current validated state (2026-09-18)

- Package/imports, registries, and both reference compositions load.
- ScenePreview rendered 240 steps with stable G1/YCB state and working camera.
- HDF5 and LeRobot ScenePreview smoke episodes were inspected and replayed.
- The previous cracker-box strategy failed three clean attempts because its
  lateral approach pushed the object without lifting it. It has been replaced
  by the sugar-box-only top-down reference task.

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
