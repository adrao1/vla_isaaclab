# Composable Isaac Lab simulation

This project is a composable Isaac Lab simulation platform. Its default scenario is a fixed-base Unitree G1 in front of a white table with an official Isaac Sim bowl and plate. The baseline controller raises and lowers both arms so the complete action path can be tested before adding grasping or learned policies.

## Environment

- Conda environment: `jiajunl_isaac`
- Environment path: `/home/vlakbnn/miniconda3/envs/jiajunl_isaac`
- Python: 3.10.21
- Isaac Sim: 4.5.0.0
- Isaac Lab: v2.0.2
- PyTorch: 2.5.1+cu121
- Tested GPU: NVIDIA GeForce RTX 4090, driver 550.144.03

Activate the isolated environment:

```bash
cd /home/vlakbnn/jiajunl4/YCB_Object
source scripts/activate.sh
check_install_environment
```

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
└── recording/            # recorder terms and compressed HDF5 backend
```

The old monolithic scene entry points were removed. `scripts/run_scenario.py` is the primary scenario runner; `scripts/replay_dataset.py` is the separate dataset replay utility. Shell scripts only activate the correct Conda environment and pass arguments to these Python entry points.

## Rates and interfaces

- Physics: 120 Hz
- Action/control: 30 Hz (`decimation=4`)
- RGB-D camera: 30 Hz, 640×480
- G1 action: 25 normalized waist, arm and hand joint commands
- Lower body: 12 joints held at the default standing target
- PickPlace policy observation: 89 values
- Reach policy observation: 81 values

## Recording and replay

Record a demonstration:

```bash
./scripts/record_scenario.sh --dataset-name g1_tabletop_raise_lower
```

Inspect and replay it:

```bash
python scripts/inspect_dataset.py outputs/datasets/g1_tabletop_raise_lower.hdf5
./scripts/replay_dataset.sh outputs/datasets/g1_tabletop_raise_lower.hdf5 --headless
```

The HDF5 file contains component IDs, rates, simulator states, actions, observations, RGB, depth, camera calibration and task distance. Tensor datasets use lightweight gzip compression. Datasets remain under `outputs/datasets/` and are ignored by Git.

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
- A future physical robot must provide its own adapter and validated joint mapping.
- Long RGB-D datasets still require disk planning even with compression.
