#!/usr/bin/env python
"""
XLeRobot Policy Client — sends observations to a remote openpi policy server
over WebSocket, receives actions, and executes them on the XLerobot hardware.

Usage:
    PYTHONPATH=src python examples/xlerobot/9_xlerobot_policy_client.py

Configuration (edit config_xlerobot.py or pass overrides):
    XLerobotConfig(id="my_xlerobot", port1="/dev/ttyUSB0", port2="/dev/ttyUSB1", ...)
"""

from __future__ import annotations

import importlib.util
import logging
import platform
import time
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client

from lerobot.robots.xlerobot import XLerobot, XLerobotConfig


# ---------------------------------------------------------------------------
# msgpack numpy serialization — mirrors openpi_client/msgpack_numpy.py
# ---------------------------------------------------------------------------

def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_packer = msgpack.Packer(default=_pack_array)


def _unpackb(data: bytes):
    return msgpack.unpackb(data, object_hook=_unpack_array, raw=False)


# ---------------------------------------------------------------------------
# WebSocket client
# ---------------------------------------------------------------------------

class PolicyClient:
    """WebSocket client that communicates with an openpi policy server."""

    def __init__(self, uri: str, *, timeout_s: float = 30.0):
        self._uri = uri
        self._timeout_s = timeout_s
        self._conn, self._metadata = self._connect()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _connect(self):
        start = time.time()
        while True:
            try:
                conn = websockets.sync.client.connect(self._uri, compression=None)
                metadata = _unpackb(conn.recv())
                logging.info("Connected to policy server at %s  metadata=%s", self._uri, metadata)
                return conn, metadata
            except (ConnectionRefusedError, OSError) as e:
                elapsed = time.time() - start
                if elapsed > self._timeout_s:
                    raise RuntimeError(f"Timed out after {self._timeout_s}s waiting for {self._uri}") from e
                logging.info("Server not ready (%s), retrying in 3s...  (%.0fs elapsed)", e, elapsed)
                time.sleep(3)

    def infer(self, obs: dict) -> dict:
        """Send observation, receive and return action dict."""
        try:
            self._conn.send(_packer.pack(obs))
            response = self._conn.recv()
        except (websockets.sync.client.ConnectionClosed, OSError):
            self.logger.warning("Server disconnected, reconnecting...")
            self._conn, self._metadata = self._connect()
            self._conn.send(_packer.pack(obs))
            response = self._conn.recv()

        if isinstance(response, str):
            raise RuntimeError(f"Policy server error:\n{response}")

        return _unpackb(response)

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# Action chunk broker — mirrors openpi_client/action_chunk_broker.py
# ---------------------------------------------------------------------------

class ActionChunkBroker:
    """Wraps a policy to return one action at a time from the chunk returned by the server.

    The policy server returns an action chunk of shape (action_horizon, action_dim).
    This broker stores the chunk and releases one step per call to get_action().
    A new inference call is only made when the current chunk is exhausted.
    """

    def __init__(self, policy_client: PolicyClient, action_horizon: int = 30):
        self._client = policy_client
        self._action_horizon = action_horizon
        self._cur_step = 0
        self._chunk: dict | None = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def _fetch_chunk(self, obs: dict) -> dict:
        """Send observation to server, return full action chunk."""
        result = self._client.infer(obs)
        actions = result.get("actions")
        if actions is None:
            raise RuntimeError(f"No 'actions' in server response: {result}")
        return result

    def get_action(self, obs: dict) -> dict:
        """Return the next action step. Fetches a new chunk from the server if needed."""
        if self._chunk is None:
            t = time.monotonic()
            self._chunk = self._fetch_chunk(obs)
            infer_ms = (time.monotonic() - t) * 1000
            self.logger.debug("Inference took %.1fms", infer_ms)
            self._cur_step = 0

        # Extract the current step from the chunk.
        # The "actions" field has shape (action_horizon, action_dim).
        # Other fields (e.g., server_timing) are passed through unchanged.
        action = {}
        for key, value in self._chunk.items():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                action[key] = value[self._cur_step]
            else:
                action[key] = value

        self._cur_step += 1
        if self._cur_step >= self._action_horizon:
            self._chunk = None

        return action

    def reset(self):
        """Reset the broker, discarding any pending chunk."""
        self._chunk = None
        self._cur_step = 0


# ---------------------------------------------------------------------------
# Zero-position initialisation — reuses SimpleTeleopArm / SimpleHeadControl
# ---------------------------------------------------------------------------

def _load_teleop_module():
    """Dynamically load helpers from 7_xlerobot_teleop_joycon.py."""
    path = Path(__file__).resolve().parent / "7_xlerobot_teleop_joycon.py"
    spec = importlib.util.spec_from_file_location("xlerobot_teleop", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tm = _load_teleop_module()
LEFT_JOINT_MAP = _tm.LEFT_JOINT_MAP
RIGHT_JOINT_MAP = _tm.RIGHT_JOINT_MAP
SimpleTeleopArm = _tm.SimpleTeleopArm
SimpleHeadControl = _tm.SimpleHeadControl
SO101Kinematics = _tm.SO101Kinematics


def init_robot(robot: XLerobot) -> None:
    """Drive all joints to zero position before starting inference."""
    obs = robot.get_observation()
    left_arm = SimpleTeleopArm(LEFT_JOINT_MAP, obs, SO101Kinematics(), prefix="left")
    right_arm = SimpleTeleopArm(RIGHT_JOINT_MAP, obs, SO101Kinematics(), prefix="right")
    head_control = SimpleHeadControl(obs)

    left_arm.move_to_zero_position(robot)
    right_arm.move_to_zero_position(robot)
    head_control.move_to_zero_position(robot)


# ---------------------------------------------------------------------------
# Observation format — mirrors openpi/policies/xlerobot_policy.py: XLeRobotInputs
# ---------------------------------------------------------------------------

ACTION_DIM = 17  # 12 arm + 2 head + 3 base

_STATE_KEYS = [
    "left_arm_shoulder_pan.pos",
    "left_arm_shoulder_lift.pos",
    "left_arm_elbow_flex.pos",
    "left_arm_wrist_flex.pos",
    "left_arm_wrist_roll.pos",
    "left_arm_gripper.pos",
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
    "head_motor_1.pos",
    "head_motor_2.pos",
    "x.vel",
    "y.vel",
    "theta.vel",
]


def format_observation(raw_obs: dict[str, Any], task: str | None = None) -> dict[str, Any]:
    """Convert a raw XLerobot observation dict into the format expected by openpi."""

    state = np.array([raw_obs.get(k, 0.0) for k in _STATE_KEYS], dtype=np.float32)

    def _parse_image(img):
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = (255 * img).astype(np.uint8)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img

    obs = {
        "observation/state": state,
        "observation/image_head": _parse_image(raw_obs["head"]),
        "observation/image_left_wrist": _parse_image(raw_obs["left_wrist"]),
        "observation/image_right_wrist": _parse_image(raw_obs["right_wrist"]),
    }
    if task:
        obs["prompt"] = task
    return obs


# ---------------------------------------------------------------------------
# Action dispatch — mirrors openpi/policies/xlerobot_policy.py: XLeRobotOutputs
# ---------------------------------------------------------------------------

def dispatch_action(action: dict, robot: XLerobot) -> None:
    """Send the action to the robot.

    The action dict contains:
        - actions: (ACTION_DIM,) step extracted from the chunk by ActionChunkBroker
        - server_timing: metadata (ignored)

    The server outputs absolute joint positions. Use directly as targets.
    """
    action_array = action.get("actions")
    if action_array is None:
        raise ValueError("No 'actions' in action dict")

    # Server returns absolute positions; use directly as targets
    formatted = {key: float(action_array[i]) for i, key in enumerate(_STATE_KEYS)}
    robot.send_action(formatted)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="XLeRobot Policy Client")
    parser.add_argument("--robot.id", default="my_xlerobot")
    parser.add_argument("--server.host", default="localhost")
    parser.add_argument("--server.port", type=int, default=8081)
    parser.add_argument("--server.task", default="Pick up the block and place it in the plate.", help="Text prompt for the policy")
    parser.add_argument("--loop.hz", type=float, default=30.0)
    parser.add_argument("--action.horizon", type=int, default=30)
    parser.add_argument("--execution.horizon", type=int, default=15, help="Number of actions to execute per chunk before fetching a new one")
    args = parser.parse_args()

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "xlerobot_policy_client.log"

    class ImmediateWriteHandler(logging.FileHandler):
        """FileHandler that flushes after every emit."""
        def emit(self, record):
            super().emit(record)
            self.flush()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(logging.StreamHandler())
    root_logger.addHandler(ImmediateWriteHandler(log_path, encoding="utf-8"))

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for h in root_logger.handlers:
        h.setFormatter(formatter)

    logger = logging.getLogger(__name__)

    if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in __import__("os").environ:
        __import__("os").environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

    # --- Robot ---
    robot_cfg = XLerobotConfig(id=getattr(args, "robot.id"))
    robot = XLerobot(robot_cfg)
    logger.info("Connecting to XLerobot...")
    robot.connect()
    logger.info("Robot connected.")

    init_robot(robot)

    # --- Policy server + action chunk broker ---
    uri = f"ws://{getattr(args, 'server.host')}:{getattr(args, 'server.port')}"
    client = PolicyClient(uri)
    broker = ActionChunkBroker(client, action_horizon=getattr(args, "action.horizon"))
    task = getattr(args, "server.task")

    # --- Inference loop ---
    loop_hz = getattr(args, "loop.hz")
    loop_period = 1.0 / loop_hz
    execution_horizon = getattr(args, "execution.horizon")
    exec_step = 0

    logger.info("Starting inference loop at %.1f Hz (execution_horizon=%d). Press Ctrl+C to stop.", loop_hz, execution_horizon)
    try:
        while True:
            t_start = time.monotonic()

            # Fetch a new action chunk from the server every execution_horizon steps
            if exec_step == 0:
                raw_obs = robot.get_observation()
                obs = format_observation(raw_obs, task=task)
                broker._chunk = broker._fetch_chunk(obs)
                # Slice chunk to only keep the first execution_horizon actions
                for key, value in broker._chunk.items():
                    if isinstance(value, np.ndarray) and value.ndim == 2:
                        broker._chunk[key] = value[:execution_horizon]
                broker._cur_step = 0
                logger.debug("Fetched new action chunk (execution_horizon=%d)", execution_horizon)

            # raw_obs = robot.get_observation()
            # obs = format_observation(raw_obs, task=task)

            # Extract the current step from the cached chunk
            chunk_action = {}
            for key, value in broker._chunk.items():
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    chunk_action[key] = value[broker._cur_step]
                else:
                    chunk_action[key] = value

            logger.info("Action: %s", chunk_action)
            dispatch_action(chunk_action, robot)

            broker._cur_step += 1
            exec_step = (exec_step + 1) % execution_horizon

            elapsed = time.monotonic() - t_start
            if elapsed > loop_period * 2:
                logger.warning("Loop iteration took %.1fms (target %.1fms)", elapsed * 1000, loop_period * 1000)

            sleep_time = loop_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        client.close()
        robot.disconnect()
        logger.info("Done.")


if __name__ == "__main__":
    main()
