"""
实时可视化 UMI 采集数据。

python scripts/vis/vis_umi.py \
    --config-path config.json \
    --tracker-device-name WM0 \
    --camera-device-id 1 \
    --gripper-serial-port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.umi import UMI
from src.pose_utils import align_tracker_pose, apply_rotation_correction_y

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("umi.vis_umi")

POSE_COLOR = [64, 200, 255]
AXIS_COLORS = {
    "x": [255, 64, 64],
    "y": [64, 255, 64],
    "z": [64, 128, 255],
}
AXIS_LENGTH = 0.03
CURRENT_AXIS_LENGTH = 0.06


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Rerun 实时可视化 UMI 采集数据")
    parser.add_argument("--config-path", type=str, default="config.json")
    parser.add_argument("--tracker-device-name", type=str, default="WM0")
    parser.add_argument("--camera-device-id", type=int, default=0)
    parser.add_argument("--gripper-serial-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--sample-hz", type=float, default=None, help="采样频率，默认按设备输出频率读取")
    parser.add_argument("--wait-for-ready-seconds", type=float, default=5.0)
    parser.add_argument("--max-history", type=int, default=600, help="轨迹保留长度，0 表示不限制")
    parser.add_argument("--application-id", type=str, default="umi.vis_umi")
    parser.add_argument("--spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rotation-correction-x-deg", type=float, default=180.0, help="X 轴额外旋转校准角度 (度)")
    parser.add_argument("--rotation-correction-y-deg", type=float, default=180.0, help="Y 轴额外旋转校准角度 (度)")
    parser.add_argument("--rotation-correction-z-deg", type=float, default=180.0, help="Z 轴额外旋转校准角度 (度)")
    return parser.parse_args()


def build_blueprint(rrb: Any) -> Any:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="Pose Trajectory", origin="/pose"),
            rrb.Vertical(
                rrb.TimeSeriesView(name="Gripper", origin="/gripper"),
                rrb.Spatial2DView(name="Frame", origin="/frame"),
                row_shares=[1, 3],
            ),
        ),
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


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    normalized = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(normalized)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("四元数范数过小，无法构建坐标系")
    qx, qy, qz, qw = normalized / norm
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float32,
    )


def get_valid_pose_components(pose_history: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    finite_mask = np.all(np.isfinite(pose_history), axis=1)
    if not np.any(finite_mask):
        return None
    poses = np.asarray(pose_history[finite_mask], dtype=np.float64)
    quaternion_norms = np.linalg.norm(poses[:, 3:], axis=1)
    poses = poses[quaternion_norms > np.finfo(np.float64).eps]
    if poses.size == 0:
        return None
    positions = np.asarray(poses[:, :3], dtype=np.float32)
    rotations = np.asarray([quaternion_to_rotation_matrix(quaternion) for quaternion in poses[:, 3:]], dtype=np.float32)
    return positions, rotations


def log_pose_axes(
    rr: Any,
    entity_path: str,
    positions: np.ndarray,
    rotations: np.ndarray,
    axis_length: float,
    radius: float,
) -> None:
    for axis_name, axis_index in (("x", 0), ("y", 1), ("z", 2)):
        axis_endpoints = positions + rotations[:, :, axis_index] * axis_length
        axis_segments = np.stack([positions, axis_endpoints], axis=1)
        rr.log(
            f"{entity_path}/{axis_name}",
            rr.LineStrips3D(axis_segments, colors=[AXIS_COLORS[axis_name]], radii=[radius]),
        )


def log_pose_trajectory(rr: Any, pose_history: np.ndarray) -> None:
    pose_components = get_valid_pose_components(pose_history)
    if pose_components is None:
        return
    positions, rotations = pose_components
    log_pose_axes(rr, "pose/tracker/history", positions, rotations, AXIS_LENGTH, 0.001)
    rr.log(
        "pose/tracker/current/origin",
        rr.Points3D(positions[-1:], colors=[POSE_COLOR], radii=[0.01]),
    )
    log_pose_axes(rr, "pose/tracker/current", positions[-1:], rotations[-1:], CURRENT_AXIS_LENGTH, 0.003)


def log_gripper(rr: Any, value: float) -> None:
    if np.isfinite(value):
        rr.log("gripper/position", rr.Scalars(float(value)))


def apply_rotation_correction_x(
    quaternion: np.ndarray,
    correction_x_deg: float = 0.0,
) -> np.ndarray:
    rot_correction = R.from_euler("x", correction_x_deg, degrees=True)
    rot_correction_inv = rot_correction.inv()
    orig_rot = R.from_quat(np.asarray(quaternion, dtype=np.float64))
    corrected_rot = rot_correction * orig_rot * rot_correction_inv
    return corrected_rot.as_quat()


def apply_rotation_correction_z(
    quaternion: np.ndarray,
    correction_z_deg: float = 0.0,
) -> np.ndarray:
    rot_correction = R.from_euler("z", correction_z_deg, degrees=True)
    rot_correction_inv = rot_correction.inv()
    orig_rot = R.from_quat(np.asarray(quaternion, dtype=np.float64))
    corrected_rot = rot_correction * orig_rot * rot_correction_inv
    return corrected_rot.as_quat()


def pose_to_array(
    pose: Any,
    rotation_correction_x_deg: float = 0.0,
    rotation_correction_y_deg: float = -90.0,
    rotation_correction_z_deg: float = 0.0,
) -> np.ndarray:
    if pose is None:
        return np.full((7,), np.nan, dtype=np.float64)
    
    position = pose.position
    rotation = pose.rotation
    
    aligned_pose = align_tracker_pose(position=position, quaternion=rotation)
    if aligned_pose is not None:
        aligned_position, aligned_quaternion, _ = aligned_pose
        position = aligned_position
        rotation = apply_rotation_correction_x(
            quaternion=aligned_quaternion,
            correction_x_deg=rotation_correction_x_deg,
        )
        rotation = apply_rotation_correction_y(
            quaternion=rotation,
            correction_y_deg=rotation_correction_y_deg,
        )
        rotation = apply_rotation_correction_z(
            quaternion=rotation,
            correction_z_deg=rotation_correction_z_deg,
        )
        
    return np.asarray([*position, *rotation], dtype=np.float64)


def wait_for_observation(umi: UMI, timeout_seconds: float) -> dict[str, Any] | None:
    deadline = time.time() + max(timeout_seconds, 0.0)
    while time.time() <= deadline:
        observation = umi.read_once(copy_frame=True)
        if observation["frame"] is not None:
            return observation
        time.sleep(0.02)
    return None


def history_container(max_history: int) -> deque[np.ndarray] | list[np.ndarray]:
    if max_history > 0:
        return deque(maxlen=max_history)
    return []


def append_history(history: deque[np.ndarray] | list[np.ndarray], pose: np.ndarray) -> None:
    history.append(np.asarray(pose, dtype=np.float64))


def history_to_array(history: deque[np.ndarray] | list[np.ndarray]) -> np.ndarray:
    if not history:
        return np.empty((0, 7), dtype=np.float64)
    return np.asarray(history, dtype=np.float64)


def log_world_origin(rr: Any) -> None:
    world_pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    world_rot = np.array([np.eye(3, dtype=np.float32)])
    log_pose_axes(rr, "pose/world_origin", world_pos, world_rot, 0.1, 0.002)
    rr.log(
        "pose/world_origin/origin",
        rr.Points3D(world_pos, colors=[[255, 255, 255]], radii=[0.005]),
    )


def visualize_live(
    application_id: str,
    spawn: bool,
    config_path: str,
    tracker_device_name: str,
    camera_device_id: int,
    gripper_serial_port: str,
    sample_hz: float | None,
    wait_for_ready_seconds: float,
    max_history: int,
    rotation_correction_x_deg: float,
    rotation_correction_y_deg: float,
    rotation_correction_z_deg: float,
) -> None:
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:
        raise RuntimeError("未安装 rerun-sdk，请先执行 `pip install -r requirements.txt`") from exc

    umi = UMI(
        tracker_config=config_path,
        tracker_device_name=tracker_device_name,
        camera_device_id=camera_device_id,
        gripper_serial_port=gripper_serial_port,
        rotation_correction_y_deg=rotation_correction_y_deg,
    )
    if not umi.connect():
        raise RuntimeError("连接 UMI 设备失败")

    try:
        first_observation = wait_for_observation(umi=umi, timeout_seconds=wait_for_ready_seconds)
        if first_observation is None:
            raise RuntimeError("未能读取到完整观测")

        rr.init(application_id, spawn=spawn)
        rr.send_blueprint(build_blueprint(rrb))
        log_world_origin(rr)

        pose_history = history_container(max_history=max_history)
        sample_interval = 1.0 / sample_hz if sample_hz and sample_hz > 0 else 0.0
        last_observation_time = -1.0
        sample_count = 0
        start_time: float | None = None
        observation = first_observation

        logger.info("开始实时可视化 UMI 数据")
        logger.info("tracker: %s, camera: %d, gripper: %s", tracker_device_name, camera_device_id, gripper_serial_port)
        logger.info("轨迹长度限制: %s", "不限制" if max_history == 0 else max_history)
        logger.info(
            "旋转校准: x=%.3f°, y=%.3f°, z=%.3f°",
            rotation_correction_x_deg,
            rotation_correction_y_deg,
            rotation_correction_z_deg,
        )

        while True:
            loop_start = time.perf_counter()
            observation_time = float(observation["time"])
            has_new_sample = observation["frame"] is not None and observation_time > last_observation_time
            if has_new_sample:
                if start_time is None:
                    start_time = observation_time
                relative_time = observation_time - start_time
                append_history(
                    pose_history,
                    pose_to_array(
                        observation["pose"],
                        rotation_correction_x_deg=rotation_correction_x_deg,
                        rotation_correction_y_deg=rotation_correction_y_deg,
                        rotation_correction_z_deg=rotation_correction_z_deg,
                    ),
                )

                rr.set_time("frame_idx", sequence=sample_count)
                rr.set_time("time", duration=relative_time)
                log_pose_trajectory(rr, history_to_array(pose_history))
                log_gripper(rr, float(np.nan if observation["gripper_pos"] is None else observation["gripper_pos"]))
                rr.log("frame/image", rr.Image(frame_to_rgb(np.asarray(observation["frame"], dtype=np.uint8))))

                sample_count += 1
                last_observation_time = observation_time

            if sample_interval > 0:
                elapsed = time.perf_counter() - loop_start
                wait_seconds = max(0.0, sample_interval - elapsed)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
            elif not has_new_sample:
                time.sleep(0.005)

            observation = umi.read_once(copy_frame=True)
    finally:
        umi.disconnect()


def run() -> int:
    args = parse_args()
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = Path(PROJECT_ROOT) / config_path
    if not config_path.exists():
        logger.error("配置文件不存在: %s", config_path)
        return 1
    if args.sample_hz is not None and args.sample_hz <= 0:
        logger.error("--sample-hz 必须大于 0: %s", args.sample_hz)
        return 1
    if args.max_history < 0:
        logger.error("--max-history 不能小于 0: %s", args.max_history)
        return 1
    try:
        visualize_live(
            application_id=args.application_id,
            spawn=args.spawn,
            config_path=str(config_path),
            tracker_device_name=args.tracker_device_name,
            camera_device_id=args.camera_device_id,
            gripper_serial_port=args.gripper_serial_port,
            sample_hz=args.sample_hz,
            wait_for_ready_seconds=args.wait_for_ready_seconds,
            max_history=args.max_history,
            rotation_correction_x_deg=args.rotation_correction_x_deg,
            rotation_correction_y_deg=args.rotation_correction_y_deg,
            rotation_correction_z_deg=args.rotation_correction_z_deg,
        )
    except KeyboardInterrupt:
        logger.info("收到中断信号，结束可视化")
        return 0
    except Exception:
        logger.exception("实时可视化失败")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
