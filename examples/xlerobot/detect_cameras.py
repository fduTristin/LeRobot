#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Detect OpenCV camera indices (or Linux /dev/video* paths), print default size/FPS/FOURCC.
# Optional: probe common resolutions; single-camera preview (--preview); all cameras live (--live).
#
# Requires: lerobot + opencv (same env as XLerobot).
#
# From LeRobot repo root:
#   set PYTHONPATH=src
#   python examples/xlerobot/detect_cameras.py
#   python examples/xlerobot/detect_cameras.py --probe-resolutions
#   python examples/xlerobot/detect_cameras.py --preview 0
#   python examples/xlerobot/detect_cameras.py --live
#
# Use listed ids in config_xlerobot.py: cam_left_wrist / cam_right_wrist / cam_head.

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

# Windows MSMF ???????????????? lerobot OpenCVCamera ????
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# ???????????????
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera  # noqa: E402


COMMON_RESOLUTIONS = (
    (640, 480),
    (848, 480),
    (1280, 720),
    (1920, 1080),
)

# opencv-python-headless has no HighGUI: cv2.imshow raises "The function is not implemented"
_OPENCV_HIGHGUI_OK: bool | None = None


def opencv_highgui_available() -> bool:
    """Return True if cv2.imshow works (needs opencv-python, not opencv-python-headless)."""
    global _OPENCV_HIGHGUI_OK
    if _OPENCV_HIGHGUI_OK is not None:
        return _OPENCV_HIGHGUI_OK
    try:
        probe = np.zeros((8, 8, 3), dtype=np.uint8)
        cv2.imshow("_detect_cameras_gui_probe", probe)
        cv2.waitKey(1)
        _OPENCV_HIGHGUI_OK = True
    except cv2.error:
        _OPENCV_HIGHGUI_OK = False
    finally:
        cv2_destroy_all_safe()
    return bool(_OPENCV_HIGHGUI_OK)


def print_headless_hint() -> None:
    print(
        "\n[display] OpenCV HighGUI is not available (common with opencv-python-headless).\n"
        "          Fix: pip uninstall opencv-python-headless opencv-python -y\n"
        "               pip install opencv-python\n"
        "          Falling back to matplotlib for live preview (close the figure window to stop).\n"
    )


def cv2_destroy_all_safe() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def probe_resolution(index_or_path: str | int, width: int, height: int) -> dict[str, Any]:
    """Open device, set requested size, read back properties and one frame."""
    cap = cv2.VideoCapture(index_or_path)
    if not cap.isOpened():
        return {"opened": False}
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, frame = cap.read()
    cap.release()
    return {
        "opened": True,
        "requested_wh": (width, height),
        "actual_wh": (aw, ah),
        "reported_fps": fps,
        "frame_ok": ok,
        "frame_shape": tuple(frame.shape) if ok and frame is not None else None,
    }


def print_cameras_table(cameras: list[dict[str, Any]]) -> None:
    if not cameras:
        print("No OpenCV cameras opened (check USB, drivers, other apps using the device).")
        return
    print(
        f"\nFound {len(cameras)} device(s). Use id below for cam_left_wrist / cam_right_wrist / cam_head.\n"
    )
    for i, info in enumerate(cameras):
        cid = info.get("id")
        prof = info.get("default_stream_profile") or {}
        print(f"--- device #{i} ---")
        print(f"  id             : {cid!r}   (type: {type(cid).__name__})")
        print(f"  backend        : {info.get('backend_api')}")
        print(f"  default size   : {prof.get('width')} x {prof.get('height')}")
        print(f"  default FPS    : {prof.get('fps')}")
        print(f"  FOURCC         : {prof.get('fourcc')!r}")
        print()


def run_probe_resolutions(cameras: list[dict[str, Any]]) -> None:
    print("\n=== Resolution probe (requested vs actual, one grab) ===\n")
    for info in cameras:
        cid = info["id"]
        print(f"device id = {cid!r}")
        for w, h in COMMON_RESOLUTIONS:
            r = probe_resolution(cid, w, h)
            if not r.get("opened"):
                print(f"  {w}x{h}  -> failed to open")
                continue
            aw, ah = r["actual_wh"]
            match = "OK" if (aw, ah) == (w, h) else "partial"
            frame = "ok" if r.get("frame_ok") else "no frame"
            print(f"  ask {w}x{h} -> got {aw}x{ah}  fps={r['reported_fps']:.2f}  [{match}]  grab:{frame}")
        print()


def run_preview(index_or_path: str | int, width: int | None, height: int | None, seconds: float) -> None:
    cap = cv2.VideoCapture(index_or_path)
    if not cap.isOpened():
        print(f"Failed to open: {index_or_path!r}")
        return
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    t_end = time.time() + seconds
    use_cv2 = opencv_highgui_available()
    if use_cv2:
        print(f"Preview {index_or_path!r}  size {aw}x{ah}  press q or ESC to quit.")
        win = "detect_cameras preview"
        try:
            while time.time() < t_end:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("read failed")
                    break
                cv2.putText(
                    frame,
                    f"{aw}x{ah}  q/ESC",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
        finally:
            cap.release()
            cv2_destroy_all_safe()
        return

    print_headless_hint()
    print(f"Preview {index_or_path!r}  size {aw}x{ah}  close the matplotlib window to stop.")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        cap.release()
        print("matplotlib is not installed; cannot show preview. Install: pip install matplotlib")
        return

    plt.ion()
    fig, ax = plt.subplots(num="detect_cameras preview")
    try:
        while time.time() < t_end:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("read failed")
                break
            ax.clear()
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{aw}x{ah}")
            ax.axis("off")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.02)
            if not plt.fignum_exists(fig.number):
                break
    finally:
        cap.release()
        plt.close(fig)
        try:
            plt.ioff()
        except Exception:
            pass


def run_live_all(
    cameras: list[dict[str, Any]],
    *,
    tile_height: int = 360,
    max_display_width: int = 1920,
    req_width: int | None = None,
    req_height: int | None = None,
    seconds: float = 600.0,
) -> None:
    """Show all cameras side-by-side in one window."""
    caps: list[cv2.VideoCapture] = []
    ids: list[str | int] = []
    for info in cameras:
        cid = info["id"]
        cap = cv2.VideoCapture(cid)
        if not cap.isOpened():
            print(f"[live] skip cannot open: {cid!r}")
            continue
        if req_width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_width)
        if req_height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_height)
        caps.append(cap)
        ids.append(cid)

    if not caps:
        print("No cameras available for live view.")
        return

    use_cv2 = opencv_highgui_available()
    print(f"[live] opened {len(caps)} stream(s), tile height {tile_height}px.")
    if use_cv2:
        print("[live] OpenCV window: q / ESC to quit.")
    else:
        print_headless_hint()

    t_end = time.time() + seconds

    def _one_frame_strip() -> np.ndarray:
        tiles: list[np.ndarray] = []
        for cap, cid in zip(caps, ids):
            ok, frame = cap.read()
            if not ok or frame is None:
                tile = np.zeros((tile_height, int(tile_height * 4 / 3), 3), dtype=np.uint8)
                cv2.putText(
                    tile,
                    "no frame",
                    (16, tile_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            else:
                h, w = frame.shape[:2]
                scale = 1.0 if h <= 0 else tile_height / float(h)
                new_w = max(1, int(w * scale))
                tile = cv2.resize(frame, (new_w, tile_height))
            label = repr(cid)
            cv2.putText(
                tile,
                label,
                (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            tiles.append(tile)

        strip = np.hstack(tiles)
        sw, sh = strip.shape[1], strip.shape[0]
        if sw > max_display_width > 0:
            scale = max_display_width / float(sw)
            nh = max(1, int(sh * scale))
            strip = cv2.resize(strip, (max_display_width, nh))
        cv2.putText(
            strip,
            "q / ESC quit" if use_cv2 else "close window to stop",
            (10, strip.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return strip

    try:
        if use_cv2:
            win = "detect_cameras LIVE"
            while time.time() < t_end:
                strip = _one_frame_strip()
                cv2.imshow(win, strip)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
        else:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                for cap in caps:
                    cap.release()
                print("matplotlib is not installed; cannot show live view. Install: pip install matplotlib")
                return

            plt.ion()
            fig, ax = plt.subplots(num="detect_cameras LIVE")
            while time.time() < t_end:
                strip = _one_frame_strip()
                ax.clear()
                ax.imshow(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
                ax.axis("off")
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.02)
                if not plt.fignum_exists(fig.number):
                    break
            plt.close(fig)
            try:
                plt.ioff()
            except Exception:
                pass
    finally:
        for cap in caps:
            cap.release()
        cv2_destroy_all_safe()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List OpenCV cameras; optional probe, single-channel or multi-channel live preview."
    )
    parser.add_argument(
        "--probe-resolutions",
        action="store_true",
        help="Try common resolutions on each device and report actual size / frame read",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Live view: stitch all detected cameras horizontally in one window",
    )
    parser.add_argument("--live-height", type=int, default=360, help="Scaled tile height (pixels)")
    parser.add_argument(
        "--live-max-width",
        type=int,
        default=1920,
        help="If stitched image is wider, scale down (0 = no limit)",
    )
    parser.add_argument("--live-width", type=int, default=None, help="Requested capture width per device")
    parser.add_argument("--live-height-cap", type=int, default=None, help="Requested capture height per device")
    parser.add_argument("--live-seconds", type=float, default=600.0, help="Max seconds for live view")
    parser.add_argument(
        "--preview",
        type=str,
        default=None,
        metavar="ID",
        help="Live preview one device only: index or path (e.g. 0 or /dev/video0)",
    )
    parser.add_argument("--preview-width", type=int, default=None)
    parser.add_argument("--preview-height", type=int, default=None)
    parser.add_argument("--preview-seconds", type=float, default=60.0, help="Max seconds for single preview")
    args = parser.parse_args()

    cameras = OpenCVCamera.find_cameras()
    print_cameras_table(cameras)

    if args.probe_resolutions and cameras:
        run_probe_resolutions(cameras)

    if args.live and cameras:
        max_w = args.live_max_width if args.live_max_width > 0 else 100000
        run_live_all(
            cameras,
            tile_height=args.live_height,
            max_display_width=max_w,
            req_width=args.live_width,
            req_height=args.live_height_cap,
            seconds=args.live_seconds,
        )

    if args.preview is not None:
        raw = args.preview.strip()
        try:
            idx: str | int = int(raw)
        except ValueError:
            idx = raw
        run_preview(idx, args.preview_width, args.preview_height, args.preview_seconds)

    if not cameras:
        sys.exit(1)


if __name__ == "__main__":
    main()
