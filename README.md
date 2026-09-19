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

The deterministic task loads only `004_sugar_box`, upright with world-Z yaw
35.81856 degrees at `(0.010509, -0.290489)` m. The Expert uses a calibrated
three-finger side grasp (hand yaw about 65 degrees, downward pitch 12 degrees),
closes all three fingers, commands an 8 cm lift, moves to a target 2 cm toward robot left from the initial box pose, levels
the box bottom, lowers, releases, and retreats. Robot left is world `-X`.

Composition:

```text
Objects-YCB-SugarBox-v0
Task-YCBPickPlaceSugarBox-v0
Expert-YCBPickPlaceSugarBox-v0
Controller-LeftArmDifferentialIK-v0
```

Run the current composition with:

```bash
./scripts/run_scenario.sh --headless --enable_cameras \
  --objects Objects-YCB-SugarBox-v0 \
  --task Task-YCBPickPlaceSugarBox-v0 \
  --expert Expert-YCBPickPlaceSugarBox-v0 \
  --controller Controller-LeftArmDifferentialIK-v0 \
  --steps 1800 \
  --preview-video outputs/videos/sugar_box_pick_place/attempt.mp4
```

The Task owns the initial/target poses, 102-dimensional physical-state
observation, reward, success, and termination. The Expert owns the deterministic
phase machine and outputs only an end-effector pose plus gripper fraction. Its
palm target uses an offset fitted to the three-finger mesh geometry and box
yaw, with a slightly downward approach. The Controller uses bounded damped least-squares
IK with persistent joint commands, a 0.025 rad command step and a 0.35 rad
tracking allowance. Saturated joints are fixed and the remaining residual is
re-solved with the free joints. Torso yaw is limited to 25 degrees. The
shared ActionTerm maps arm and hand targets to 25 normalized actions.

Current status: the sugar-box recording run triggered Task success at step
961 after grasp, left transport, leveling, lowering, opening, and withdrawal.
The user-approved success tolerance is now 15 mm XY (previously 8 mm); palm
separation must exceed 20 cm to exclude a held object. Height and speed limits
and the 15-frame hold are retained. The successful episode contains 961 frames
at 30 Hz, with terminal and success flags set on the last frame.
Dataset: `outputs/lerobot/sugar_box_left_place_20260919/`.
Recording diagnostics: `outputs/videos/sugar_box_lerobot_20260919/`.
The earlier release preview is in `outputs/videos/sugar_box_release_20260919/`.

The base starts at `(0, -0.64, 0.74)` m and the open left palm at approximately
`(-0.1765, -0.4943, 0.7780)` m. Generated outputs are ignored by Git.

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

### Record a sugar-box demonstration

```bash
cd /home/vlakbnn/jiajunl4/isaac_simlab
./scripts/run_scenario.sh --headless \
  --objects Objects-YCB-SugarBox-v0 \
  --task Task-YCBPickPlaceSugarBox-v0 \
  --expert Expert-YCBPickPlaceSugarBox-v0 \
  --controller Controller-LeftArmDifferentialIK-v0 \
  --record-format lerobot --episodes 1 --steps 1500 \
  --dataset-name sugar_box_left_place_demo

./scripts/replay_lerobot.sh outputs/lerobot/sugar_box_left_place_demo \
  --episode 0 --headless
```

`--steps` is the maximum number of 30 Hz control steps per episode; successful
termination ends it earlier. `--episodes` selects the number of episodes.
Use a new dataset name for each run; existing datasets are not overwritten.
The runner saves only successful sugar-box demonstrations and aborts export
if an episode fails. Success requires XY error <15 mm, height error <15 mm,
linear speed <0.04 m/s, angular speed <0.30 rad/s, and palm-to-box distance
>20 cm for 15 consecutive steps. This tolerance was deliberately relaxed from
8 mm XY after the release trajectory left an 8.165 mm placement error.

The output `outputs/lerobot/<dataset-name>/` uses LeRobot v3 at 30 FPS,
including front RGB video, 25 joint positions and velocities, 25 joint target
actions in radians, normalized simulator actions, object/end-effector state,
reward, terminal/success flags, task text, seed, and phase index. Depth is not
exported by this adapter. A temporary HDF5 staging file is converted automatically.
The convenience `record_lerobot.sh` defaults to a stationary scene preview and
its prompt; use the explicit command above for sugar-box collection.

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
