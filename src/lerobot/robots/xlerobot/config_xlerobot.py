# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig


def make_xlerobot_opencv_cameras(
    left_index_or_path: str | int,
    right_index_or_path: str | int,
    head_index_or_path: str | int,
    *,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    async_read_timeout_ms: float = 500.0,
) -> dict[str, CameraConfig]:
    """Three OpenCV cameras for XLerobot (left wrist, right wrist, head).

    `index_or_path` can be an integer device index (0, 1, 2) or a path string
    (e.g. ``/dev/video0`` on Linux or ``0`` on Windows as str).
    """
    return {
        "left_wrist": OpenCVCameraConfig(
            index_or_path=left_index_or_path,
            fps=fps,
            width=width,
            height=height,
            rotation=Cv2Rotation.NO_ROTATION,
            async_read_timeout_ms=async_read_timeout_ms,
        ),
        "right_wrist": OpenCVCameraConfig(
            index_or_path=right_index_or_path,
            fps=fps,
            width=width,
            height=height,
            rotation=Cv2Rotation.NO_ROTATION,
            async_read_timeout_ms=async_read_timeout_ms,
        ),
        "head": OpenCVCameraConfig(
            index_or_path=head_index_or_path,
            fps=fps,
            width=width,
            height=height,
            rotation=Cv2Rotation.NO_ROTATION,
            async_read_timeout_ms=async_read_timeout_ms,
        ),
    }


def xlerobot_cameras_config() -> dict[str, CameraConfig]:
    """Empty placeholder; use :class:`XLerobotConfig` camera port fields + ``__post_init__``."""
    return {}


@RobotConfig.register_subclass("xlerobot")
@dataclass
class XLerobotConfig(RobotConfig):

    port1: str = "COM4"  # port to connect to the bus (so101 + head camera)
    port2: str = "COM5"  # port to connect to the bus (same as lekiwi setup)
    calibration_dir: Path = Path(r"E:\FVL\EmbodiedAI\XLeRobot\LeRobot\data\calibration")
    
    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # OpenCV camera device index or path (override per machine). Used when `cameras` is left empty.
    cam_left_wrist: str | int = 0
    cam_right_wrist: str | int = 2
    cam_head: str | int = 3
    cam_fps: int = 30
    cam_width: int = 640
    cam_height: int = 480
    # OpenCV async_read wait; raise if USB/MSMF is slow (e.g. 200 ms was too tight on Windows).
    cam_async_read_timeout_ms: float = 500.0

    # Joy-Con dataset recording (examples/xlerobot/7_xlerobot_teleop_joycon_record.py)
    record_data_root: Path = Path(r"E:\FVL\EmbodiedAI\XLeRobot\LeRobot\data\recordings")  # None -> HF_LEROBOT_HOME at runtime
    record_exp_name: str = "pick_and_place"
    record_task: str = "xlerobot joycon task"
    record_repo_id: str | None = None  # None -> "local/{record_exp_name}"
    record_resume: bool = True
    record_overwrite: bool = False
    record_push_to_hub: bool = False
    record_show_preview: bool = True
    record_preview_height: int = 240
    # Same idea as detect_cameras.py --live-max-width; 0 = no downscale.
    record_preview_max_width: int = 1920

    cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "i",
            "backward": "k",
            "left": "j",
            "right": "l",
            "rotate_left": "u",
            "rotate_right": "o",
            # Speed control
            "speed_up": "n",
            "speed_down": "m",
            # quit teleop
            "quit": "b",
        }
    )

    def __post_init__(self):
        if not self.cameras:
            object.__setattr__(
                self,
                "cameras",
                make_xlerobot_opencv_cameras(
                    self.cam_left_wrist,
                    self.cam_right_wrist,
                    self.cam_head,
                    fps=self.cam_fps,
                    width=self.cam_width,
                    height=self.cam_height,
                    async_read_timeout_ms=self.cam_async_read_timeout_ms,
                ),
            )
        super().__post_init__()


@dataclass
class XLerobotHostConfig:
    # Network Configuration
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # Duration of the application
    connection_time_s: int = 3600

    # Watchdog: stop the robot if no command is received for over 0.5 seconds.
    watchdog_timeout_ms: int = 500

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30

@RobotConfig.register_subclass("xlerobot_client")
@dataclass
class XLerobotClientConfig(RobotConfig):
    # Network Configuration
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "i",
            "backward": "k",
            "left": "j",
            "right": "l",
            "rotate_left": "u",
            "rotate_right": "o",
            # Speed control
            "speed_up": "n",
            "speed_down": "m",
            # quit teleop
            "quit": "b",
        }
    )

    cam_left_wrist: str | int = 0
    cam_right_wrist: str | int = 2
    cam_head: str | int = 3
    cam_fps: int = 30
    cam_width: int = 640
    cam_height: int = 480
    cam_async_read_timeout_ms: float = 500.0

    cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5

    def __post_init__(self):
        if not self.cameras:
            object.__setattr__(
                self,
                "cameras",
                make_xlerobot_opencv_cameras(
                    self.cam_left_wrist,
                    self.cam_right_wrist,
                    self.cam_head,
                    fps=self.cam_fps,
                    width=self.cam_width,
                    height=self.cam_height,
                    async_read_timeout_ms=self.cam_async_read_timeout_ms,
                ),
            )
        super().__post_init__()
