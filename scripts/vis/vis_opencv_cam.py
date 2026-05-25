"""
实时可视化 OpenCVCam 画面。

python scripts/vis/vis_opencv_cam.py \
    --camera-device 0 \
    --camera-width 1080 \
    --camera-height 640 \
    --camera-fps 30
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.devices.opencv_cam import OpenCVCam

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("umi.vis_opencv_cam")


def parse_camera_device(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Rerun 实时可视化 OpenCVCam")
    parser.add_argument("--camera-device", type=str, default="0", help="OpenCV 相机索引或视频路径")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--read-thread-fps", type=int, default=120)
    parser.add_argument("--sample-hz", type=float, default=None, help="可视化刷新频率，默认按设备输出频率读取")
    parser.add_argument("--wait-for-ready-seconds", type=float, default=5.0)
    parser.add_argument("--application-id", type=str, default="umi.vis_opencv_cam")
    parser.add_argument("--spawn", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_blueprint(rrb: object) -> object:
    return rrb.Blueprint(
        rrb.Spatial2DView(name="OpenCVCam", origin="/camera"),
        collapse_panels=True,
    )


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim != 3:
        raise ValueError(f"frame 维度异常: {frame.shape}")
    if frame.shape[2] == 3:
        return np.ascontiguousarray(frame[..., ::-1])
    if frame.shape[2] == 4:
        return np.ascontiguousarray(frame[..., [2, 1, 0, 3]])
    raise ValueError(f"不支持的 frame 通道数: {frame.shape[2]}")


def wait_for_frame(camera: OpenCVCam, timeout_seconds: float) -> tuple[np.ndarray, float] | None:
    deadline = time.time() + max(timeout_seconds, 0.0)
    while time.time() <= deadline:
        frame, frame_time = camera.read_once(copy_frame=True, return_time=True)
        if frame is not None:
            return np.asarray(frame), float(frame_time)
        time.sleep(0.02)
    return None


def visualize_live(
    application_id: str,
    spawn: bool,
    camera_device: str,
    camera_width: int | None,
    camera_height: int | None,
    camera_fps: int,
    read_thread_fps: int,
    sample_hz: float | None,
    wait_for_ready_seconds: float,
) -> None:
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:
        raise RuntimeError("未安装 rerun-sdk，请先执行 `pip install -r requirements.txt`") from exc

    camera = OpenCVCam(
        device_id=parse_camera_device(camera_device),
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fps=camera_fps,
        read_thread_fps=read_thread_fps,
    )
    if not camera.connect():
        raise RuntimeError(f"连接 OpenCVCam 失败: {camera_device}")

    try:
        first_frame = wait_for_frame(camera=camera, timeout_seconds=wait_for_ready_seconds)
        if first_frame is None:
            raise RuntimeError("未能读取到相机画面")

        rr.init(application_id, spawn=spawn)
        rr.send_blueprint(build_blueprint(rrb))

        sample_interval = 1.0 / sample_hz if sample_hz and sample_hz > 0 else 0.0
        frame, frame_time = first_frame
        last_frame_time = -1.0
        frame_idx = 0
        start_time: float | None = None

        logger.info(
            "开始可视化 OpenCVCam: device=%s, width=%s, height=%s, fps=%s, read_thread_fps=%s",
            camera_device,
            camera_width,
            camera_height,
            camera_fps,
            read_thread_fps,
        )

        while True:
            loop_start = time.perf_counter()
            has_new_frame = frame is not None and float(frame_time) > last_frame_time
            if has_new_frame:
                current_frame_time = float(frame_time) if frame_time > 0 else time.time()
                if start_time is None:
                    start_time = current_frame_time
                rr.set_time("frame_idx", sequence=frame_idx)
                rr.set_time("time", duration=current_frame_time - start_time)
                rr.log("camera/image", rr.Image(frame_to_rgb(np.asarray(frame, dtype=np.uint8))))
                rr.log(
                    "camera/info",
                    rr.TextDocument(
                        "\n".join(
                            [
                                f"device: {camera_device}",
                                f"frame_idx: {frame_idx}",
                                f"frame_time: {current_frame_time:.6f}",
                                f"shape: {tuple(int(x) for x in frame.shape)}",
                            ]
                        )
                    ),
                )
                frame_idx += 1
                last_frame_time = current_frame_time

            if sample_interval > 0:
                elapsed = time.perf_counter() - loop_start
                wait_seconds = max(0.0, sample_interval - elapsed)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
            elif not has_new_frame:
                time.sleep(0.005)

            next_frame = camera.read_once(copy_frame=True, return_time=True)
            frame, frame_time = next_frame
    finally:
        camera.disconnect()


def run() -> int:
    args = parse_args()
    if args.sample_hz is not None and args.sample_hz <= 0:
        logger.error("--sample-hz 必须大于 0: %s", args.sample_hz)
        return 1
    if args.camera_width is not None and args.camera_width <= 0:
        logger.error("--camera-width 必须大于 0: %s", args.camera_width)
        return 1
    if args.camera_height is not None and args.camera_height <= 0:
        logger.error("--camera-height 必须大于 0: %s", args.camera_height)
        return 1
    if args.camera_fps <= 0:
        logger.error("--camera-fps 必须大于 0: %s", args.camera_fps)
        return 1
    if args.read_thread_fps <= 0:
        logger.error("--read-thread-fps 必须大于 0: %s", args.read_thread_fps)
        return 1

    camera_device = args.camera_device
    if not camera_device.isdigit():
        camera_path = Path(camera_device)
        if not camera_path.is_absolute():
            camera_path = Path(PROJECT_ROOT) / camera_path
        camera_device = str(camera_path)

    try:
        visualize_live(
            application_id=args.application_id,
            spawn=args.spawn,
            camera_device=camera_device,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
            read_thread_fps=args.read_thread_fps,
            sample_hz=args.sample_hz,
            wait_for_ready_seconds=args.wait_for_ready_seconds,
        )
    except KeyboardInterrupt:
        logger.info("收到中断信号，结束可视化")
        return 0
    except Exception:
        logger.exception("OpenCVCam 可视化失败")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
