from __future__ import annotations

# Windows Joy-Con HID: use BetterJoy + correct hidapi stack; do not mix package `hid` with `hidapi`.
# See: joycon-robotics/hidapi_for_windows/README_hidapi.md

import importlib.util
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path
from typing import cast

# Match detect_cameras.py: avoid MSMF transform issues on some Windows + OpenCV builds.
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.model.SO101Robot import SO101Kinematics
from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say


def _load_joycon_teleop_module():
    path = Path(__file__).resolve().parent / "7_xlerobot_teleop_joycon.py"
    spec = importlib.util.spec_from_file_location("xlerobot_joycon_teleop", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_jm = _load_joycon_teleop_module()
FixedAxesJoyconRobotics = _jm.FixedAxesJoyconRobotics
SimpleTeleopArm = _jm.SimpleTeleopArm
SimpleHeadControl = _jm.SimpleHeadControl
HEAD_MOTOR_MAP = _jm.HEAD_MOTOR_MAP
LEFT_JOINT_MAP = _jm.LEFT_JOINT_MAP
RIGHT_JOINT_MAP = _jm.RIGHT_JOINT_MAP
get_joycon_base_action = _jm.get_joycon_base_action
get_joycon_speed_control = _jm.get_joycon_speed_control


class RecordingFixedAxesJoyconRobotics(FixedAxesJoyconRobotics):
    """Same as FixedAxesJoyconRobotics, but Plus/Minus drive dataset recording pulses instead of reset."""

    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)
        self.record_start_pulse = False
        self.record_stop_pulse = False

    def common_update(self):
        # Same race as FixedAxesJoyconRobotics: background thread can run before __init__ sets stick centers.
        if not hasattr(self, "joycon_stick_v_0"):
            if self.joycon.is_right():
                self.joycon_stick_v_0 = 1900
                self.joycon_stick_h_0 = 2100
            else:
                self.joycon_stick_v_0 = 2300
                self.joycon_stick_h_0 = 2000
        if not hasattr(self, "last_gripper_button_state"):
            self.gripper_speed = 0.4
            self.gripper_direction = 1
            self.gripper_min = 0
            self.gripper_max = 90
            self.last_gripper_button_state = 0

        # Do NOT clear record_* here: common_update runs in the joycon background thread (~100 Hz) while the
        # main loop runs at cam_fps (~30 Hz). Clearing every frame made the Plus/Minus pulses vanish before
        # the main thread could see them. Main loop snapshots and clears after handling.
        speed_scale = 0.001

        orientation_rad = self.get_orientation()
        roll, pitch, yaw = orientation_rad

        joycon_stick_v = self.joycon.get_stick_right_vertical() if self.joycon.is_right() else self.joycon.get_stick_left_vertical()
        joycon_stick_v_threshold = 300
        joycon_stick_v_range = 1000
        if joycon_stick_v > joycon_stick_v_threshold + self.joycon_stick_v_0:
            self.position[0] += (
                speed_scale
                * (joycon_stick_v - self.joycon_stick_v_0)
                / joycon_stick_v_range
                * self.dof_speed[0]
                * self.direction_reverse[0]
                * math.cos(pitch)
            )
            self.position[2] += (
                speed_scale
                * (joycon_stick_v - self.joycon_stick_v_0)
                / joycon_stick_v_range
                * self.dof_speed[1]
                * self.direction_reverse[1]
                * math.sin(pitch)
            )
        elif joycon_stick_v < self.joycon_stick_v_0 - joycon_stick_v_threshold:
            self.position[0] += (
                speed_scale
                * (joycon_stick_v - self.joycon_stick_v_0)
                / joycon_stick_v_range
                * self.dof_speed[0]
                * self.direction_reverse[0]
                * math.cos(pitch)
            )
            self.position[2] += (
                speed_scale
                * (joycon_stick_v - self.joycon_stick_v_0)
                / joycon_stick_v_range
                * self.dof_speed[1]
                * self.direction_reverse[1]
                * math.sin(pitch)
            )

        joycon_stick_h = self.joycon.get_stick_right_horizontal() if self.joycon.is_right() else self.joycon.get_stick_left_horizontal()
        joycon_stick_h_threshold = 300
        joycon_stick_h_range = 1000
        if joycon_stick_h > joycon_stick_h_threshold + self.joycon_stick_h_0:
            self.position[1] += (
                speed_scale
                * (joycon_stick_h - self.joycon_stick_h_0)
                / joycon_stick_h_range
                * self.dof_speed[1]
                * self.direction_reverse[1]
            )
        elif joycon_stick_h < self.joycon_stick_h_0 - joycon_stick_h_threshold:
            self.position[1] += (
                speed_scale
                * (joycon_stick_h - self.joycon_stick_h_0)
                / joycon_stick_h_range
                * self.dof_speed[1]
                * self.direction_reverse[1]
            )

        joycon_button_up = self.joycon.get_button_r() if self.joycon.is_right() else self.joycon.get_button_l()
        if joycon_button_up == 1:
            self.position[2] += speed_scale * self.dof_speed[2] * self.direction_reverse[2]

        joycon_button_down = self.joycon.get_button_r_stick() if self.joycon.is_right() else self.joycon.get_button_l_stick()
        if joycon_button_down == 1:
            self.position[2] -= speed_scale * self.dof_speed[2] * self.direction_reverse[2]

        joycon_button_home = self.joycon.get_button_home() if self.joycon.is_right() else self.joycon.get_button_capture()
        if joycon_button_home == 1:
            self.position = self.offset_position_m.copy()

        for event_type, status in self.button.events():
            if self.joycon.is_right() and event_type == "plus" and status == 1:
                self.record_start_pulse = True
            elif self.joycon.is_left() and event_type == "minus" and status == 1:
                self.record_stop_pulse = True
            elif self.joycon.is_right() and event_type == "a":
                self.next_episode_button = status
            elif self.joycon.is_right() and event_type == "y":
                self.restart_episode_button = status
            else:
                self.reset_button = 0

        gripper_button_pressed = False
        if self.joycon.is_right():
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zr() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_r_btn() == 1
        else:
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zl() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_l_btn() == 1

        if gripper_button_pressed and self.last_gripper_button_state == 0:
            self.gripper_direction *= -1
            print(f"[GRIPPER] Direction changed to: {'Open' if self.gripper_direction == 1 else 'Close'}")

        self.last_gripper_button_state = gripper_button_pressed

        if gripper_button_pressed:
            new_gripper_state = self.gripper_state + self.gripper_direction * self.gripper_speed
            if new_gripper_state >= self.gripper_min and new_gripper_state <= self.gripper_max:
                self.gripper_state = new_gripper_state

        if self.joycon.is_right():
            if self.next_episode_button == 1:
                self.button_control = 1
            elif self.restart_episode_button == 1:
                self.button_control = -1
            elif self.reset_button == 1:
                self.button_control = 8
            else:
                self.button_control = 0

        return self.position, self.gripper_state, self.button_control


def main() -> None:
    cfg = XLerobotConfig(id="my_xlerobot")
    fps = cfg.cam_fps

    data_root = Path(cfg.record_data_root).expanduser() if cfg.record_data_root else HF_LEROBOT_HOME
    dataset_dir = data_root / cfg.record_exp_name
    data_root.mkdir(parents=True, exist_ok=True)
    repo_id = cfg.record_repo_id or f"local/{cfg.record_exp_name}"
    info_json = dataset_dir / "meta" / "info.json"

    robot = XLerobot(cfg)
    joycon_right: RecordingFixedAxesJoyconRobotics | None = None
    joycon_left: RecordingFixedAxesJoyconRobotics | None = None

    try:
        robot.connect()
        print("[MAIN] Connected to robot.")
    except Exception as e:
        print(f"[MAIN] Failed to connect: {e}")
        return

    # Dataset must be resolved before opening Joy-Cons: sys.exit/abort used to leave HID threads running.
    features: dict = {}
    features.update(hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=True))
    features.update(hw_to_dataset_features(cast(dict[str, type | tuple], robot.action_features), ACTION, use_video=True))

    dataset: LeRobotDataset | None = None
    if cfg.record_resume and dataset_dir.is_dir() and info_json.is_file():
        try:
            dataset = LeRobotDataset(repo_id, root=dataset_dir, batch_encoding_size=1)
        except FileNotFoundError as e:
            # create() only writes info.json until the first save_episode(); resume then fails on missing
            # tasks.parquet / meta/episodes/*.parquet. Other cases: partial or corrupted meta/.
            total_eps = -1
            try:
                with open(info_json, encoding="utf-8") as f:
                    total_eps = int(json.load(f).get("total_episodes", -1))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
            cause = e.__cause__ if e.__cause__ is not None else e
            if total_eps == 0:
                print(
                    f"[RECORD] Cannot resume: metadata incomplete (typical if no episode was saved yet). "
                    f"Detail: {cause}\n"
                    f"  Removing {dataset_dir} and creating a new dataset."
                )
                shutil.rmtree(dataset_dir)
                dataset = None
            else:
                print(
                    f"[RECORD] Cannot resume dataset: {cause}\n"
                    f"  Repair files under {dataset_dir}, or set record_overwrite=True to remove and start fresh."
                )
                robot.disconnect()
                return
        if dataset is not None:
            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(0, 4 * len(robot.cameras))
            print(f"[RECORD] Resumed dataset: {dataset_dir} (episodes already saved: {dataset.meta.total_episodes})")
    elif cfg.record_resume and dataset_dir.is_dir() and not info_json.is_file():
        print(
            f"[RECORD] Folder exists but no LeRobot dataset at {info_json}. "
            "Removing empty/incomplete folder and creating a new dataset."
        )
        shutil.rmtree(dataset_dir)
    elif dataset_dir.exists() and not cfg.record_resume:
        if cfg.record_overwrite:
            shutil.rmtree(dataset_dir)
        else:
            print(f"[RECORD] Dataset directory exists: {dataset_dir}")
            print("  Set record_resume=True or record_overwrite=True in XLerobotConfig (config_xlerobot.py).")
            robot.disconnect()
            return
    if dataset is None:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=dataset_dir,
            use_videos=True,
            robot_type=robot.name,
            image_writer_processes=0,
            image_writer_threads=4 * len(robot.cameras),
            batch_encoding_size=1,
        )
        print(f"[RECORD] Created dataset at {dataset.root}")

    try:
        recording = False

        joycon_right = RecordingFixedAxesJoyconRobotics("right", dof_speed=[2, 2, 2, 1, 1, 1])
        joycon_left = RecordingFixedAxesJoyconRobotics("left", dof_speed=[2, 2, 2, 1, 1, 1])

        obs = robot.get_observation()
        kin_left = SO101Kinematics()
        kin_right = SO101Kinematics()
        left_arm = SimpleTeleopArm(LEFT_JOINT_MAP, obs, kin_left, prefix="left", speed_config=None)
        right_arm = SimpleTeleopArm(RIGHT_JOINT_MAP, obs, kin_right, prefix="right", speed_config=None)
        head_control = SimpleHeadControl(obs, speed_config=None)

        left_arm.move_to_zero_position(robot)
        right_arm.move_to_zero_position(robot)
        head_control.move_to_zero_position(robot)

        while True:
            start_loop_t = time.perf_counter()
            pose_right, gripper_right, control_button_right = joycon_right.get_control()
            pose_left, gripper_left, control_button_left = joycon_left.get_control()

            # Latch: background thread sets these; we clear after reading so pulses survive thread speed mismatch.
            record_start = joycon_right.record_start_pulse
            if record_start:
                joycon_right.record_start_pulse = False
            record_stop = joycon_left.record_stop_pulse
            if record_stop:
                joycon_left.record_stop_pulse = False

            if record_start and not recording:
                recording = True
                ep_idx = dataset.meta.total_episodes
                log_say(f"Recording started. Episode index {ep_idx}", blocking=False)
                print(f"[RECORD] START recording episode_index={ep_idx}")

            if record_stop and recording:
                recording = False

                # Use episode_buffer["size"] instead of dataset.num_frames
                # dataset.num_frames returns the number of SAVED frames on disk,
                # not the frames in the current episode buffer waiting to be saved
                n_frames = dataset.episode_buffer["size"]

                if n_frames > 0:
                    dataset.save_episode()
                    done_idx = dataset.meta.total_episodes - 1
                    log_say(f"Episode saved. Index {done_idx}.", blocking=False)
                    print(f"[RECORD] STOP: Saved ep {done_idx}. Total now: {dataset.meta.total_episodes}")
                else:
                    print("[RECORD] STOP: No frames to save.")

            if control_button_right == 8:
                print("[MAIN] Reset to zero position!")
                right_arm.move_to_zero_position(robot)
                left_arm.move_to_zero_position(robot)
                continue

            # Get current observation (before executing action)
            obs = robot.get_observation()

            # Record current observation (state before action execution)
            if recording:
                obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)

            # Compute and send action
            right_arm.target_positions["gripper"] = gripper_right
            left_arm.target_positions["gripper"] = gripper_left

            right_arm.handle_joycon_input(pose_right, gripper_right)
            right_action = right_arm.p_control_action(robot, obs=obs)
            left_arm.handle_joycon_input(pose_left, gripper_left)
            left_action = left_arm.p_control_action(robot, obs=obs)
            head_action = head_control.p_control_action(robot, obs=obs)

            base_action = get_joycon_base_action(joycon_right, robot)
            speed_multiplier = get_joycon_speed_control(joycon_right)
            if base_action:
                for key in base_action:
                    if "vel" in key or "velocity" in key:
                        base_action[key] *= speed_multiplier

            action = {**left_action, **right_action, **head_action, **base_action}
            sent_action = robot.send_action(action)

            # Record action (paired with the prior observation)
            if recording:
                action_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
                frame = {**obs_frame, **action_frame, "task": cfg.record_task}
                dataset.add_frame(frame)

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(1.0 / fps - dt_s, 0.0))

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted.")
    finally:
        if dataset is not None:
            dataset.finalize()
            if cfg.record_push_to_hub:
                dataset.push_to_hub()
        if joycon_right is not None:
            joycon_right.disconnect()
        if joycon_left is not None:
            joycon_left.disconnect()
        robot.disconnect()
        print("[MAIN] Done.")


if __name__ == "__main__":
    main()
