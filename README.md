# isaac_simlab

`isaac_simlab` is a small, composable Isaac Lab framework for shared manipulation
research. It keeps task meaning, scripted strategy, and robot control separate.

```text
Scenario
├── World
├── Robot
├── Objects
├── Sensors
├── Task
├── Expert       (optional)
├── Controller
└── Recorder     (optional)
```

For scripted manipulation, the runtime path is:

```text
Task state -> Expert -> EE pose + gripper target -> Controller
          -> normalized action -> env.step() -> Isaac Lab ActionTerm -> robot
```

- **Task**: observations, reward, success, failure, and goal definition.
- **Expert**: task-specific scripted phases and desired end-effector/gripper targets.
- **Controller**: reusable IK/control that produces normalized environment actions.
- **Robot**: asset, joints, limits, end effectors, and reset pose.
- **Recorder**: physical observations, actions, camera data, and simulator metadata.

There is deliberately no Command, Policy, or Action Adapter layer. Shared
normalized-action conversion remains in `actions.py` and the Isaac Lab
`ActionTerm`.

## Environment

- Conda environment: `jiajunl_isaac`
- Python 3.10.21
- Isaac Sim 4.5.0.0
- Isaac Lab v2.0.2
- PyTorch 2.5.1+cu121
- LeRobot 0.4.3 / Dataset v3

Always activate the project environment before running Python:

```bash
cd /home/vlakbnn/jiajunl4/isaac_simlab
source scripts/activate.sh
check_install_environment
```

## Source layout

```text
src/isaac_simlab/
├── contracts.py
├── registry.py
├── scenario.py
├── runtime.py
├── actions.py
├── worlds/
├── robots/
├── objects/
├── sensors/
├── tasks/
├── experts/
├── controllers/
└── recording/
```

List all registered components:

```bash
./scripts/run_scenario.sh --headless --list-components
```

## Reference scenarios

### Scene preview

This scenario needs no Expert. The Standing Controller maps the G1 default
physical joint targets to the normalized 25-dimensional environment action.

```bash
./scripts/run_scenario.sh \
  --headless \
  --world World-Tabletop-v0 \
  --robot Robot-UnitreeG1-v0 \
  --objects Objects-YCB-Basic-v0 \
  --sensors Sensors-FixedRGBD-v0 \
  --task Task-ScenePreview-v0 \
  --controller Controller-Standing-v0 \
  --steps 240 \
  --preview-video outputs/previews/isaac_simlab_scene_preview.mp4
```

The YCB set contains `003_cracker_box`, `004_sugar_box`,
`005_tomato_soup_can`, and `006_mustard_bottle`. This validates the G1, white
table, objects, RGB-D camera, registry, scenario composition, and normalized
action path.

### Sugar-box pick and place

The deterministic task loads only `004_sugar_box`. The left hand first rises,
moves horizontally above the object, rotates its fingers downward, descends,
grasps, moves the box 2 cm toward robot right, and places it on the table. In
this setup robot right is base `-Y`, which is world `+X`.

Composition:

```text
Objects-YCB-SugarBox-v0
Task-YCBPickPlaceSugarBox-v0
Expert-YCBPickPlaceSugarBox-v0
Controller-LeftArmDifferentialIK-v0
```

The Task owns the initial/target poses, 102-dimensional physical-state
observation, reward, success, and termination. The Expert owns the deterministic
phase machine and outputs only an end-effector pose plus gripper fraction. The
Controller uses Isaac Lab `DifferentialIKController`, limits joint deltas and
joint positions, merges the hand target, and uses the shared ActionTerm mapping
to produce a normalized 25-dimensional action.

The bounded runner uses a fresh Isaac process for every attempt, stops after at
most three failures, and writes videos plus diagnostics under
`outputs/videos/sugar_box_pick_place/`. No failed attempt is recorded as a
successful demonstration.

## Actions and rates

- Physics: 120 Hz
- Control and RGB-D camera: 30 Hz
- Environment action: 25 normalized controlled-joint commands
- Observed joint position/velocity: physical radians and radians/second
- The ActionTerm converts normalized actions back to physical joint targets

## Recording and replay

HDF5 remains the raw/debug format; LeRobot v3 is the training-oriented format.
Both preserve physical robot state separately from normalized simulator action.

```bash
./scripts/record_scenario.sh --dataset-name scenario_smoke
./scripts/replay_dataset.sh outputs/datasets/scenario_smoke.hdf5 --headless

./scripts/record_lerobot.sh --dataset-name lerobot_smoke
./scripts/replay_lerobot.sh outputs/lerobot/lerobot_smoke --episode 0 --headless
```

Validated refactor smoke artifacts are available at
`outputs/datasets/isaac_simlab_scene_preview_smoke.hdf5` and
`outputs/lerobot/isaac_simlab_scene_preview_smoke/`; both were inspected and
replayed successfully.

## Contributor workflow

```text
Need a new task                 -> add a Task
Need scripted demonstrations   -> add an Expert
Existing controller fits       -> reuse it
Need a new control modality    -> add a reusable Controller
```

Tasks must not contain scripted phases, IK, or joint control. Experts must not
compute Jacobians, run differential IK, or normalize actions. Controllers must
not contain object semantics, task phases, reward, or success logic.
