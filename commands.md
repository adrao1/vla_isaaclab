# 常用命令

```bash
cd <isaac_simlab-checkout>
conda activate <developer-env>
source scripts/activate.sh
check_install_environment
```

查看注册组件：

```bash
./scripts/run_scenario.sh --headless --list-components
```

预览 G1、白桌、四个 YCB 物体和 RGB-D 相机：

```bash
./scripts/run_scenario.sh --headless \
  --objects Objects-YCB-Basic-v0 \
  --task Task-ScenePreview-v0 \
  --controller Controller-Standing-v0 \
  --steps 240 \
  --preview-video outputs/previews/isaac_simlab_scene_preview.mp4
```

HDF5 录制、检查与回放：

```bash
./scripts/record_scenario.sh --dataset-name scenario_smoke
python scripts/inspect_dataset.py outputs/datasets/scenario_smoke.hdf5
./scripts/replay_dataset.sh outputs/datasets/scenario_smoke.hdf5 --headless
```

LeRobot v3 录制、检查与回放：

```bash
./scripts/record_lerobot.sh --dataset-name lerobot_smoke
python scripts/inspect_lerobot.py outputs/lerobot/lerobot_smoke
./scripts/replay_lerobot.sh outputs/lerobot/lerobot_smoke --episode 0 --headless
```
