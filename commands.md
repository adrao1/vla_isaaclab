# 常用命令

```bash
cd <vla_isaaclab-checkout>
conda activate <developer-env>
source scripts/activate.sh
check_install_environment
```

查看注册环境：

```bash
./scripts/run_env.sh --headless --physics-only --list-tasks
```

预览 G1、白桌、四个 YCB 物体和 RGB-D 相机：

```bash
./scripts/run_env.sh --headless \
  --task VLA-ScenePreview-YCB-G1-v0 \
  --steps 240 \
  --preview-video outputs/previews/vla_isaaclab_scene_preview.mp4
```

HDF5 录制与检查：

```bash
./scripts/record_env.sh --dataset-name environment_smoke
python scripts/inspect_dataset.py outputs/datasets/environment_smoke.hdf5
```

Sugar-box 成功演示录制为 LeRobot v3 数据集（1 个 episode，最多 1200 步）：

```bash
./scripts/record_lerobot.sh \
  --dataset-name sugar_box_demo \
  --episodes 1 \
  --steps 1200
```

数据保存在 `outputs/lerobot/sugar_box_demo/`；检查和回放：

```bash
python scripts/inspect_lerobot.py outputs/lerobot/sugar_box_demo
./scripts/replay_lerobot.sh outputs/lerobot/sugar_box_demo --episode 0 --headless
```
