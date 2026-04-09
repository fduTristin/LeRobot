# LeRobot for XLeRobot project

> Customized for XLeRobot project on Windows.

## 代码下载与环境配置

### LeRobot

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg=7.1.1 -c conda-forge -y
git clone https://github.com/fduTristin/LeRobot.git
cd LeRobot
pip install -e ".[feetech]"
```

### Joycon-Robotics

参考 [joycon-robotics](https://github.com/box2ai-robotics/joycon-robotics/blob/main/README.md) 的 `README.md` 进行代码下载与环境配置。

## 使用 Swich Joycon 遥操作

### 连接 Joycon

1. 同时按住左边手柄侧面 SL 和 SR 之间的圆形按钮和右边手柄侧面的圆形按钮至左右指示灯闪烁。
2. 在主机蓝牙分别连接 Joy-Con(L) 和 Joy-Con(R)。若连接成功，两个手柄的指示灯常亮。

### 测试连接

```bash
# 之后默认所有代码都在 lerobot 环境中运行
conda activate lerobot
cd LeRobot
python examples/xlerobot/joycon_test_read_CN.py
```

成功示例：

```text
Fixed-axis test (Windows / hidapi)
Vertical stick: X (forward/back); horizontal stick: Y (left/right)
L: Z+ ; stick click: Z- ; Capture: reset pose; ZL: gripper toggle
Ctrl+C to stop

[OK] Connected left Joy-Con (L)
   Manufacturer: Nintendo
   Product: Wireless Gamepad
✅ IMU完整初始化成功
[OK] Joy-Con read loop started
请将JoyCon平放在桌面...
开始校准，请保持JoyCon静止...
收集陀螺仪偏移数据...
✅ 校准完成！陀螺仪偏移: [-7.70132569e-04  3.91592832e-04  5.60969785e-01]
✅ 初始姿态：Roll=279.7° Pitch=0.4°
pos_left=0.000, 0.000, 0.000, Rot_left=3.311, -0.012, 0.000, gripper_left=0.5, control=0
```

### 遥操

TODO
