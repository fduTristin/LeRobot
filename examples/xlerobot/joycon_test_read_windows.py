# Fixed-axis Joy-Con readout test for Windows (hidapi path; does not use joyconrobotics hid).

"""
Equivalent to joycon_test_read_CN.py on Windows: fixed-axis mapping without joyconrobotics hid.

Requires: pip install hidapi numpy
Setup: ViGEmBus, pair Joy-Con over Bluetooth, run BetterJoy (see
    joycon-robotics/hidapi_for_windows/README_hidapi.md)

Run from repo root or this folder (script adds hidapi_for_windows to sys.path):
    python joycon_test_read_windows.py

Default is left Joy-Con (same as joycon_test_read_CN.py). Set DEVICE = "right" below if you only have a right Joy-Con.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

# Add joycon-robotics/hidapi_for_windows so we can import JoyConHIDAPIReader
_REPO = Path(__file__).resolve().parents[3]  # .../XLeRobot
_HIDAPI_DIR = _REPO / "joycon-robotics" / "hidapi_for_windows"
if not _HIDAPI_DIR.is_dir():
    raise RuntimeError(f"hidapi_for_windows folder not found: {_HIDAPI_DIR}")
sys.path.insert(0, str(_HIDAPI_DIR))

# hid must come from pip package `hidapi`; PyPI package `hid` conflicts and fails to load hidapi.dll on Windows
try:
    import hid  # noqa: F401
except ImportError as e:
    if "Unable to load" in str(e) or "libraries" in str(e).lower():
        print(
            "[ERROR] Failed to load HID native library. The pip package `hid` conflicts with `hidapi`.\n"
            "In your current conda env run:\n"
            "  pip uninstall hid -y\n"
            "  pip install hidapi\n"
            "See joycon-robotics/hidapi_for_windows/README_hidapi.md\n",
            file=sys.stderr,
        )
    raise

from joycon_hidapi_reader import JoyConHIDAPIReader  # noqa: E402

# Same vendor/PIDs as joyconrobotics/constants.py
_JOYCON_VENDOR = 0x057E
_PID_LEFT = 0x2006


class JoyConHIDAPIReaderLeft(JoyConHIDAPIReader):
    """Left Joy-Con: open PID 0x2006; parse left stick bytes 6-8 and left-hand buttons."""

    def connect(self):
        import hid

        try:
            self.device = hid.device()
            self.device.open(_JOYCON_VENDOR, _PID_LEFT)
            self.device.set_nonblocking(1)
            print("[OK] Connected left Joy-Con (L)")
            print(f"   Manufacturer: {self.device.get_manufacturer_string()}")
            print(f"   Product: {self.device.get_product_string()}")
            if not self._enable_imu():
                print("[WARN] IMU init failed; gyro may not work")
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            print("[OK] Joy-Con read loop started")
            return True
        except Exception as e:
            print(f"[ERR] Connect failed: {e}")
            return False

    def _parse_input_report(self, data):
        """Left Joy-Con buttons and stick; same byte layout as Linux joycon.py for report 0x30."""
        with self.lock:
            buttons_left = data[5]
            buttons_shared = data[4]
            # Byte 5: matches joycon.py get_button_* for left Joy-Con
            self.buttons = {
                "Y": False,
                "X": False,
                "B": False,
                "A": False,
                "HOME": False,
                "CAPTURE": bool(buttons_shared & 0x20),
                "STICK": bool(buttons_shared & 0x08),  # stick click
                "UP": bool(buttons_left & 0x02),
                "DOWN": bool(buttons_left & 0x01),
                "L": bool(buttons_left & 0x40),  # bit 6: L, Z+
                "ZL": bool(buttons_left & 0x80),  # bit 7: ZL, gripper toggle
            }
            # Left stick raw, bytes 6-8
            stick_raw = data[6] | ((data[7] & 0x0F) << 8)
            stick_x = (stick_raw - 2048) / 2048.0
            stick_y_raw = (data[7] >> 4) | (data[8] << 4)
            stick_y = (stick_y_raw - 2048) / 2048.0
            self.stick_x = np.clip(stick_x, -1.0, 1.0)
            self.stick_y = np.clip(stick_y, -1.0, 1.0)

            imu_offset = 13
            accel_x_raw = int.from_bytes(data[imu_offset : imu_offset + 2], "little", signed=True)
            accel_y_raw = int.from_bytes(data[imu_offset + 2 : imu_offset + 4], "little", signed=True)
            accel_z_raw = int.from_bytes(data[imu_offset + 4 : imu_offset + 6], "little", signed=True)
            gyro_x_raw = int.from_bytes(data[imu_offset + 6 : imu_offset + 8], "little", signed=True)
            gyro_y_raw = int.from_bytes(data[imu_offset + 8 : imu_offset + 10], "little", signed=True)
            gyro_z_raw = int.from_bytes(data[imu_offset + 10 : imu_offset + 12], "little", signed=True)

            ACCEL_SCALE = 4096.0
            self.accel[0] = accel_x_raw / ACCEL_SCALE
            self.accel[1] = accel_y_raw / ACCEL_SCALE
            self.accel[2] = accel_z_raw / ACCEL_SCALE
            GYRO_SCALE = 13.371
            self.gyro[0] = (gyro_x_raw / GYRO_SCALE) * (np.pi / 180.0)
            self.gyro[1] = (gyro_y_raw / GYRO_SCALE) * (np.pi / 180.0)
            self.gyro[2] = (gyro_z_raw / GYRO_SCALE) * (np.pi / 180.0)
            self.gyro -= self.gyro_offset
            self._update_attitude()


class FixedAxesWindowsController:
    """
    Mirrors FixedAxesJoyconRobotics.common_update plus pose output (same idea as
    joyconrobotic_hidapi.JoyConController: roll offset, negated scaled pitch).
    """

    def __init__(
        self,
        reader: JoyConHIDAPIReader,
        *,
        is_left: bool,
        joycon_stick_v_0: int = 2300,
        joycon_stick_h_0: int = 2000,
        dof_speed=None,
        direction_reverse=None,
        offset_position_m=None,
        pitch_gain: float = 1.5,
    ):
        self.reader = reader
        self.is_left = is_left
        self.joycon_stick_v_0 = joycon_stick_v_0
        self.joycon_stick_h_0 = joycon_stick_h_0
        self.dof_speed = dof_speed or [2, 2, 2, 1, 1, 1]
        self.direction_reverse = direction_reverse or [1, 1, 1]
        self.offset_position_m = list(offset_position_m or [0.0, 0.0, 0.0])
        self.position = self.offset_position_m.copy()
        self.init_position = self.position.copy()
        self.pitch_gain = pitch_gain

        self.gripper_open = 0.5
        self.gripper_close = -0.15
        self.gripper_state = self.gripper_open
        self.gripper_toggle_button = 0
        self.last_zl = False

        self.next_episode_button = 0
        self.restart_episode_button = 0
        self.reset_button = 0
        self.button_control = 0

    def _raw_from_norm(self, norm: float, center: int) -> int:
        """Map normalized stick [-1, 1] to a raw-like scale for the same thresholds as the CN script."""
        return int(center + norm * 1000)

    def get_control(self):
        state = self.reader.get_state()
        bt = state["buttons"]

        # Map normalized sticks to raw-like values (same thresholds as joycon_test_read_CN.py)
        if self.is_left:
            raw_v = self._raw_from_norm(float(state["stick_y"]), self.joycon_stick_v_0)
            raw_h = self._raw_from_norm(float(state["stick_x"]), self.joycon_stick_h_0)
        else:
            raw_v = self._raw_from_norm(float(state["stick_y"]), self.joycon_stick_v_0)
            raw_h = self._raw_from_norm(float(state["stick_x"]), self.joycon_stick_h_0)

        th, rg = 300, 1000

        if raw_v > th + self.joycon_stick_v_0:
            self.position[0] += 0.001 * (raw_v - self.joycon_stick_v_0) / rg * self.dof_speed[0]
        elif raw_v < self.joycon_stick_v_0 - th:
            self.position[0] += 0.001 * (raw_v - self.joycon_stick_v_0) / rg * self.dof_speed[0]

        if raw_h > th + self.joycon_stick_h_0:
            self.position[1] += (
                0.001
                * (raw_h - self.joycon_stick_h_0)
                / rg
                * self.dof_speed[1]
                * self.direction_reverse[1]
            )
        elif raw_h < self.joycon_stick_h_0 - th:
            self.position[1] += (
                0.001
                * (raw_h - self.joycon_stick_h_0)
                / rg
                * self.dof_speed[1]
                * self.direction_reverse[1]
            )

        if self.is_left:
            if bt.get("L"):
                self.position[2] += 0.001 * self.dof_speed[2]
            if bt.get("STICK"):
                self.position[2] -= 0.001 * self.dof_speed[2]
            if bt.get("UP"):
                self.position[0] += 0.001 * self.dof_speed[0]
            if bt.get("DOWN"):
                self.position[0] -= 0.001 * self.dof_speed[0]
            if bt.get("CAPTURE"):
                self.position = self.offset_position_m.copy()
        else:
            if bt.get("R"):
                self.position[2] += 0.001 * self.dof_speed[2]
            if bt.get("STICK"):
                self.position[2] -= 0.001 * self.dof_speed[2]
            if bt.get("X"):
                self.position[0] += 0.001 * self.dof_speed[0]
            if bt.get("B"):
                self.position[0] -= 0.001 * self.dof_speed[0]
            if bt.get("HOME"):
                self.position = self.offset_position_m.copy()

        # Gripper: ZL (left) / ZR (right) rising edge, same idea as original zl/zr handler
        zg = bt.get("ZL") if self.is_left else bt.get("ZR")
        if zg and not self.last_zl:
            self.gripper_state = self.gripper_close if self.gripper_state == self.gripper_open else self.gripper_open
        self.last_zl = bool(zg)

        roll = state["roll"]
        pitch = state["pitch"]
        yaw = state["yaw"]
        roll = roll - np.pi / 2
        pitch = -pitch * self.pitch_gain
        pose = self.position + [roll, pitch, yaw]

        self.button_control = 0
        return pose, self.gripper_state, self.button_control

    def disconnect(self):
        self.reader.disconnect()


def main():
    # "left" | "right"
    DEVICE = "left"

    print("Fixed-axis test (Windows / hidapi)")
    print("Vertical stick: X (forward/back); horizontal stick: Y (left/right)")
    if DEVICE == "left":
        print("L: Z+ ; stick click: Z- ; Capture: reset pose; ZL: gripper toggle")
    else:
        print("R: Z+ ; stick click: Z- ; Home: reset pose; ZR: gripper toggle")
    print("Ctrl+C to stop\n")

    reader = JoyConHIDAPIReaderLeft() if DEVICE == "left" else JoyConHIDAPIReader()
    if not reader.connect():
        return
    reader.calibrate(samples=100)

    joycon_stick_v_0 = 2300 if DEVICE == "left" else 1900
    joycon_stick_h_0 = 2000 if DEVICE == "left" else 2100

    ctrl = FixedAxesWindowsController(
        reader,
        is_left=(DEVICE == "left"),
        joycon_stick_v_0=joycon_stick_v_0,
        joycon_stick_h_0=joycon_stick_h_0,
        dof_speed=[2, 2, 2, 1, 1, 1],
    )

    try:
        for _ in range(10000):
            pose, gripper, button_control = ctrl.get_control()
            x, y, z, roll, pitch, yaw = pose
            side = "left" if DEVICE == "left" else "right"
            print(
                f"pos_{side}={x:.3f}, {y:.3f}, {z:.3f}, "
                f"Rot_{side}={roll:.3f}, {pitch:.3f}, {yaw:.3f}, "
                f"gripper_{side}={gripper}, control={button_control}"
            )
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ctrl.disconnect()


if __name__ == "__main__":
    main()
