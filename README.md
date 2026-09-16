# Composable Isaac Lab simulation

This project is a composable Isaac Lab simulation platform. Its default scenario is a fixed-base Unitree G1 in front of a white table with an official Isaac Sim bowl and plate. The baseline controller raises and lowers both arms so the complete action path can be tested before adding grasping or learned policies.

## Environment

- Conda environment: `jiajunl_isaac`
- Environment path: `/home/vlakbnn/miniconda3/envs/jiajunl_isaac`
- Python: 3.10.21
- Isaac Sim: 4.5.0.0
- Isaac Lab: v2.0.2
- PyTorch: 2.5.1+cu121
- LeRobot: 0.4.3 (Dataset v3.0)
- Tested GPU: NVIDIA GeForce RTX 4090, driver 550.144.03

Activate the isolated environment:

```bash
cd /home/vlakbnn/jiajunl4/YCB_Object
source scripts/activate.sh
check_install_environment
```

The compatible LeRobot writer dependencies are already installed. To reproduce
that installation in this same Conda environment, run:

```bash
./scripts/install_lerobot_dataset.sh
```

The script checks `python`, `pip`, and `conda info --envs` before every install
command. It pins the Isaac-compatible PyTorch stack and installs LeRobot with
`--no-deps`, preventing pip from replacing Isaac Sim's PyTorch packages.

## Run the default scenario

Headless validation with the arm raise/lower controller:

```bash
./scripts/validate_scenario.sh
```

Open the GUI until the window is closed:

```bash
./scripts/run_scenario.sh --steps 0
```

The default composition is:

```text
World-Tabletop-v0
Robot-UnitreeG1-v0
Objects-Dinnerware-v0
Sensors-FixedRGBD-v0
Task-PickPlace-v0
Controller-RaiseLower-v0
```

The current `PickPlace` task defines observations, distance reward and success conditions. The baseline controller only raises and lowers the arms; it does not attempt a grasp.

## Composition architecture

```text
Scenario
├── World       geometry, semantic frames and regions
├── Robot       asset, joint groups and end-effector names
├── Objects     physical assets and task roles
├── Sensors     world- or robot-mounted sensor rigs
├── Task        observations, rewards and termination conditions
├── Controller  scripted, IK, teleoperation or policy action source
└── Recorder    actions, observations, state, sensor data and metadata
```

The registry validates component capabilities before creating the Isaac Lab environment. The composer then produces one `ManagerBasedRLEnvCfg`; tasks do not contain table prim paths or G1-specific joint names.

Current registered components:

| Kind | IDs |
|---|---|
| World | `World-Tabletop-v0`, `World-Pedestal-v0` |
| Robot | `Robot-UnitreeG1-v0` |
| Objects | `Objects-Dinnerware-v0`, `Objects-YCB-Basic-v0`, `Objects-None-v0` |
| Sensors | `Sensors-FixedRGBD-v0` |
| Tasks | `Task-PickPlace-v0`, `Task-Reach-v0` |
| Controllers | `Controller-Standing-v0`, `Controller-RaiseLower-v0` |

List them from the executable registry:

```bash
./scripts/run_scenario.sh --headless --list-components
```

Example alternative composition:

```bash
./scripts/run_scenario.sh \
  --headless \
  --world World-Pedestal-v0 \
  --objects Objects-None-v0 \
  --task Task-Reach-v0 \
  --controller Controller-Standing-v0 \
  --steps 300
```

## Source layout

```text
src/sim_platform/
├── contracts.py          # component interfaces
├── registry.py           # component IDs and lookup
├── scenario.py           # compatibility checks and environment composition
├── runtime.py            # common ManagerBasedRLEnv runtime
├── actions.py            # normalized robot action term
├── worlds/               # tabletop and pedestal geometry/semantics
├── robots/               # Unitree G1 adapter
├── objects/              # dinnerware, YCB and empty object sets
├── sensors/              # fixed RGB-D rig
├── tasks/                # PickPlace and Reach manager configurations
├── controllers/          # standing and arm raise/lower baselines
└── recording/            # frame adapter, HDF5 staging and LeRobot v3 writer
```

The old monolithic scene entry points were removed. `scripts/run_scenario.py` is
the primary scenario runner. HDF5 and LeRobot each have separate inspection and
replay utilities. Shell scripts activate and verify the correct Conda environment
before passing arguments to Python.

## Rates and interfaces

- Physics: 120 Hz
- Action/control: 30 Hz (`decimation=4`)
- RGB-D camera: 30 Hz, 640×480
- G1 action: 25 normalized waist, arm and hand joint commands
- Lower body: 12 joints held at the default standing target
- PickPlace policy observation: 89 values
- Reach policy observation: 81 values

## LeRobot recording pipeline

Record the default scenario as a local LeRobot Dataset v3 dataset. The default
command records two episodes with 120 frames per episode:

```bash
./scripts/record_lerobot.sh
```

The resulting dataset is written to:

```text
outputs/lerobot/g1_dinnerware_raise_lower/
├── data/                         # Parquet state and action records
├── videos/                       # 640x480, 30 Hz AV1 front-camera video
└── meta/                         # features, episodes, tasks, stats and simulation manifest
```

Inspect the dataset and decode representative video frames:

```bash
python scripts/inspect_lerobot.py outputs/lerobot/g1_dinnerware_raise_lower
```

Replay episode 0 through Isaac Sim and save a new camera image:

```bash
./scripts/replay_lerobot.sh \
  outputs/lerobot/g1_dinnerware_raise_lower \
  --episode 0 \
  --headless
```

The replay output is `outputs/replay/lerobot_rgb.png`, with a JSON report beside
it. Recording and replay use a neutral temporary file between the LeRobot process
and Isaac Sim process. This process boundary avoids loading PyAV/TorchVision and
Omniverse native libraries into the same Python process.

Recorded fields:

| Field | Shape | Meaning |
|---|---:|---|
| `observation.state` | 25 | G1 controlled-joint positions in radians |
| `observation.velocity` | 25 | G1 controlled-joint velocities in rad/s |
| `action` | 25 | physical joint-position targets in radians |
| `sim.action.normalized` | 25 | simulator action used for exact replay |
| `observation.environment_state` | 28 | two end-effector poses and two object poses |
| `observation.images.front` | 3x480x640 | RGB camera video |
| `next.reward`, `next.done`, `next.success` | 1 | transition outcome |
| `sim.seed` | 1 | episode reset seed |

Custom recordings can select any registered scenario components:

```bash
./scripts/run_scenario.sh \
  --headless \
  --record-format lerobot \
  --episodes 10 \
  --steps 300 \
  --dataset-name my_dataset \
  --task-prompt "Raise and lower both arms."
```

Datasets are local and are not uploaded to Hugging Face Hub.

## Native HDF5 recording

The original compressed HDF5 backend remains useful for debugging RGB-D and raw
Isaac Lab recorder output:

```bash
./scripts/record_scenario.sh --dataset-name g1_tabletop_raise_lower
```

Inspect and replay it:

```bash
python scripts/inspect_dataset.py outputs/datasets/g1_tabletop_raise_lower.hdf5
./scripts/replay_dataset.sh outputs/datasets/g1_tabletop_raise_lower.hdf5 --headless
```

The HDF5 file contains component IDs, rates, simulator states, actions,
observations, RGB, depth, camera calibration and task distance. Tensor datasets
use lightweight gzip compression. Generated datasets remain under `outputs/`
and are ignored by Git.

## Assets and physics

- Robot: Isaac Lab v2.0.2 Unitree G1 asset
- Dinnerware: Isaac Sim 4.5 ArchVis `bowl_plate.usd` and `plate_large.usd`
- YCB: official Isaac Sim 4.5 `Axis_Aligned_Physics` assets cached locally
- Dinnerware visuals use explicit rigid bodies, masses and hidden cylinder collision proxies
- The bowl proxy models external support and does not model its concave interior

Asset inspection tools remain available:

```bash
python scripts/inspect_assets.py
python scripts/inspect_dinnerware_assets.py
```

## Adding components

- World: create a `WorldDefinition`, publish semantic spawn/target regions and register it in `worlds/__init__.py`.
- Robot: create a `RobotDefinition` with joint groups, end effectors and an Isaac Lab articulation configuration.
- Task: define manager configurations and required capabilities, without hard-coded world prim paths or robot joint names.
- Object set: expose entities through roles such as `manipulation_object` and `goal`.
- Controller: produce the action vector exposed by the selected robot action term.
- Sensor rig: use stable sensor names so recording consumers do not depend on prim paths.

## Current limitations

- The G1 root is fixed; balance and locomotion are outside the current task scope.
- The arm raise/lower controller only validates the control interface.
- PickPlace does not yet include IK, grasp phases or gripper logic.
- The current LeRobot sample records the raise/lower control baseline; it is not
  a successful object pick-and-place demonstration.
- Depth remains available in the HDF5 backend but is not part of the current
  LeRobot RGB schema.
- A future physical robot must provide its own adapter and validated joint mapping.
- Long RGB-D datasets still require disk planning even with compression.
