# 项目命令

## 激活环境

```bash
cd /home/vlakbnn/jiajunl4/isaaclab_manipulation
source scripts/activate.sh
check_install_environment
```

## 默认场景

默认组合是 G1、白色桌面、碗、盘子、固定 RGB-D 相机和双臂抬起/放下控制器。

碗和盘子位于靠机器人一侧的桌沿附近。

无界面运行 300 个控制步：

```bash
./scripts/validate_scenario.sh
```

打开 GUI 并持续运行：

```bash
./scripts/run_scenario.sh --steps 0
```

保持默认站姿，不进行手臂动作：

```bash
./scripts/run_scenario.sh --headless --controller Controller-Standing-v0 --steps 300
```

## 组合场景、任务和对象

查看注册组件：

```bash
./scripts/run_scenario.sh --headless --list-components
```

在桌面场景使用 YCB 对象：

```bash
./scripts/run_scenario.sh \
  --headless \
  --objects Objects-YCB-Basic-v0 \
  --task Task-PickPlace-v0 \
  --steps 300
```

使用同一个 Reach 任务切换到 Pedestal 场景：

```bash
./scripts/run_scenario.sh \
  --headless \
  --world World-Pedestal-v0 \
  --objects Objects-None-v0 \
  --task Task-Reach-v0 \
  --controller Controller-Standing-v0 \
  --steps 300
```

### 微波炉桌面场景

运行 G1、白色桌面和微波炉场景，机器人保持站立，微波炉门保持关闭：

```bash
./scripts/run_scenario.sh \
  --headless \
  --world World-MicrowaveTabletop-v0 \
  --objects Objects-Microwave-v0 \
  --task Task-ScenePreview-v0 \
  --controller Controller-Standing-v0 \
  --steps 240 \
  --preview-video outputs/previews/microwave_tabletop_standing.mp4
```

重新转换和检查已经随项目保存的微波炉 MJCF（约 1.5 MB，无额外下载）：

```bash
./scripts/prepare_microwave_asset.sh
python scripts/inspect_microwave_asset.py --headless
```

### YCB 物体场景预览

预览 cracker box、sugar box、tomato soup can 和 mustard bottle：

```bash
./scripts/run_scenario.sh \
  --headless \
  --world World-Tabletop-v0 \
  --objects Objects-YCB-Basic-v0 \
  --task Task-ScenePreview-v0 \
  --controller Controller-Standing-v0 \
  --steps 240 \
  --preview-video outputs/previews/ycb_scene_preview.mp4
```

## 数据录制与回放

### LeRobot Dataset v3

默认录制 2 个 episode，每个 120 帧：

```bash
./scripts/record_lerobot.sh
```

自定义 episode 数量和长度（后面的参数会覆盖脚本默认值）：

```bash
./scripts/record_lerobot.sh \
  --episodes 10 \
  --steps 300 \
  --dataset-name my_dataset \
  --task-prompt "Raise and lower both arms."
```

检查正式数据集，包括代表性视频帧解码：

```bash
python scripts/inspect_lerobot.py outputs/lerobot/g1_ycb_dinnerware_raise_lower
```

在 Isaac Sim 中回放第 0 个 episode：

```bash
./scripts/replay_lerobot.sh \
  outputs/lerobot/g1_ycb_dinnerware_raise_lower \
  --episode 0 \
  --headless
```

如需在同一 `jiajunl_isaac` 环境中重新安装兼容的数据集依赖：

```bash
./scripts/install_lerobot_dataset.sh
```

### 原生 HDF5

录制 120 个控制步：

```bash
./scripts/record_scenario.sh --dataset-name g1_tabletop_raise_lower
```

短录制测试：

```bash
./scripts/run_scenario.sh \
  --headless \
  --record-format hdf5 \
  --steps 4 \
  --dataset-name scenario_smoke
```

检查和回放：

```bash
python scripts/inspect_dataset.py outputs/datasets/scenario_smoke.hdf5
./scripts/replay_dataset.sh outputs/datasets/scenario_smoke.hdf5 --headless
```

## 输出位置

```text
outputs/scenarios/<scenario-id>/rgb.png
outputs/scenarios/<scenario-id>/validation.json
outputs/datasets/*.hdf5
outputs/lerobot/<dataset-name>/
outputs/replay/rgb.png
outputs/replay/rgb.json
outputs/replay/lerobot_rgb.png
outputs/replay/lerobot_rgb.json
```

## 资产检查

```bash
./scripts/prepare_ycb_dinnerware.sh
python scripts/inspect_assets.py
python scripts/inspect_dinnerware_assets.py
```

## 查看资源

```bash
df -h /home/vlakbnn/jiajunl4
nvidia-smi
```
