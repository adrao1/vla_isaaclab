# 项目命令

## 激活环境

```bash
cd /home/vlakbnn/jiajunl4/YCB_Object
source scripts/activate.sh
check_install_environment
```

## 默认场景

默认组合是 G1、白色桌面、碗、盘子、固定 RGB-D 相机和双臂抬起/放下控制器。

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

## 数据录制与回放

录制 120 个控制步：

```bash
./scripts/record_scenario.sh --dataset-name g1_tabletop_raise_lower
```

短录制测试：

```bash
./scripts/run_scenario.sh --headless --record --steps 4 --dataset-name scenario_smoke
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
outputs/replay/rgb.png
outputs/replay/rgb.json
```

## 资产检查

```bash
python scripts/inspect_assets.py
python scripts/inspect_dinnerware_assets.py
```

## 查看资源

```bash
df -h /home/vlakbnn/jiajunl4
nvidia-smi
```
