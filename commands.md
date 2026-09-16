# 项目命令

项目目录：

```bash
cd /home/vlakbnn/jiajunl4/YCB_Object
```

## 激活独立 Conda 环境

```bash
source scripts/activate.sh
```

该脚本激活默认 Conda 目录中的环境：

```text
/home/vlakbnn/miniconda3/envs/jiajunl_isaac
```

确认当前 Python、pip 和 Conda 环境均属于 `jiajunl_isaac`：

```bash
check_install_environment
```

也可以分别查看：

```bash
which python
which pip
conda info --envs
python --version
```

## 运行自动验证

无界面运行 1200 个物理步，验证碰撞、落体稳定性、质量和惯量，同时保存相机图像：

```bash
./scripts/validate_scene.sh
```

验证结果位于：

```text
outputs/physics-validation.json
outputs/scene_rgb.png
outputs/ycb_pick_place_scene.usd
outputs/validate-scene.log
```

验证时不保存 RGB 图片：

```bash
./scripts/validate_scene.sh --no-image
```

## 启动场景

用 GUI 启动场景并运行 600 个物理步：

```bash
./scripts/run_scene.sh --enable_cameras --steps 600
```

保持 GUI 运行，直到手动关闭窗口：

```bash
./scripts/run_scene.sh --enable_cameras --steps 0
```

服务器无显示器时，以无界面方式运行：

```bash
./scripts/run_scene.sh --headless --enable_cameras --steps 1200
```

选择 CUDA 设备：

```bash
./scripts/run_scene.sh --headless --enable_cameras --device cuda:0 --steps 1200
```

## 运行 G1、碗和盘子场景

无界面运行 1200 个物理步，固定 Unitree G1 基座并保持站立，验证餐具落桌稳定性并保存截图：

```bash
./scripts/validate_dinnerware_scene.sh
```

持续打开 GUI，直到手动关闭窗口：

```bash
./scripts/run_dinnerware_scene.sh --steps 0
```

在普通 SSH 或 VS Code Remote 中运行有限步数：

```bash
./scripts/run_dinnerware_scene.sh --headless --steps 1200
```

对应输出：

```text
outputs/dinnerware_g1_scene_rgb.png
outputs/dinnerware_g1_scene.usd
outputs/dinnerware-g1-validation.json
outputs/validate-dinnerware-g1-scene.log
```

G1 使用 Isaac Lab v2.0.2 官方 `G1_CFG`。机器人根部固定，腿部保持默认站立目标；`torso_joint`（waist yaw）、双肩、双肘和手指关节仍由执行器驱动。

餐具资产静态检查：

```bash
source scripts/activate.sh
python scripts/inspect_dinnerware_assets.py
```

## 接入机器人 USD

机器人型号确定后，通过绝对路径加载机器人 USD：

```bash
./scripts/run_scene.sh --enable_cameras --robot-usd /absolute/path/to/robot.usd --steps 0
```

无界面验证机器人 USD 能否随场景加载：

```bash
./scripts/run_scene.sh --headless --enable_cameras --robot-usd /absolute/path/to/robot.usd --steps 1200
```

未提供 `--robot-usd` 时，场景会保留 `/World/RobotSpawn` 接口，不会假设机器人型号。

## 检查 YCB 资产

先激活环境，再运行静态资产检查：

```bash
source scripts/activate.sh
python scripts/inspect_assets.py
```

该脚本检查配置中的 4 个 YCB 物体，包括：

- USD 能否加载
- rigid body
- collision
- 质量
- 包围盒、单位和坐标轴

检查结果写入：

```text
outputs/asset-inspection.json
```

如需同时保存完整终端日志：

```bash
python scripts/inspect_assets.py > outputs/inspect-physics-assets.log 2>&1
```

## 查看输出

查看物理验证报告：

```bash
python -m json.tool outputs/physics-validation.json
```

查看资产检查报告：

```bash
python -m json.tool outputs/asset-inspection.json
```

确认截图文件：

```bash
file outputs/scene_rgb.png
file outputs/dinnerware_scene_rgb.png
file outputs/dinnerware_g1_scene_rgb.png
```

查看项目及磁盘占用：

```bash
du -sh /home/vlakbnn/jiajunl4/YCB_Object
df -h /home/vlakbnn/jiajunl4
```

## 查看脚本参数

```bash
./scripts/run_scene.sh --help
```

主要参数：

- `--headless`：无界面运行
- `--enable_cameras`：启用相机渲染
- `--steps N`：运行 N 个物理步；GUI 下设为 0 表示持续运行
- `--robot-usd PATH`：加载指定机器人 USD
- `--no-image`：不保存 RGB 图片
- `--device cuda:0`：选择仿真设备
