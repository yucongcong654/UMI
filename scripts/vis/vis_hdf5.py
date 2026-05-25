"""
Visualize UMI hdf5 data.
python scripts/vis/vis_hdf5.py data/2026_04_13_2.hdf5 --realtime
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.pose_utils import align_tracker_pose, apply_rotation_correction_y

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("umi.vis_hdf5")

POSE_DATASETS = (("pose", "tracker"), ("left_pose", "left"), ("right_pose", "right"))
POSE_COLORS = {
    "tracker": [64, 200, 255],
    "left": [255, 64, 64],
    "right": [64, 128, 255],
}
AXIS_COLORS = {
    "x": [255, 64, 64],
    "y": [64, 255, 64],
    "z": [64, 128, 255],
}
AXIS_LENGTH = 0.03
CURRENT_AXIS_LENGTH = 0.06


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Rerun 可视化 UMI hdf5 数据")
    parser.add_argument("hdf5_path", type=str, help="UMI 录制生成的 .h5 或 .hdf5 文件路径")
    parser.add_argument("--application-id", type=str, default="umi.vis_hdf5")
    parser.add_argument("--spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="按数据时间戳节奏发送日志，获得播放效果",
    )
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=1.0,
        help="播放倍率，1.0 为原速，2.0 为两倍速",
    )
    parser.add_argument("--rotation-correction-x-deg", type=float, default=180.0, help="X 轴额外旋转校准角度 (度)")
    parser.add_argument("--rotation-correction-y-deg", type=float, default=180.0, help="Y 轴额外旋转校准角度 (度)")
    parser.add_argument("--rotation-correction-z-deg", type=float, default=180.0, help="Z 轴额外旋转校准角度 (度)")
    return parser.parse_args()


def require_dataset(h5_file: h5py.File, dataset_name: str) -> h5py.Dataset:
    if dataset_name not in h5_file:
        raise KeyError(f"hdf5 文件缺少数据集: {dataset_name}")
    return h5_file[dataset_name]


def get_pose_datasets(h5_file: h5py.File) -> list[tuple[str, h5py.Dataset]]:
    pose_datasets: list[tuple[str, h5py.Dataset]] = []
    for dataset_name, pose_name in POSE_DATASETS:
        if dataset_name in h5_file:
            pose_datasets.append((pose_name, h5_file[dataset_name]))
    if not pose_datasets:
        raise KeyError("hdf5 文件缺少 pose 数据集")
    return pose_datasets


def validate_shapes(
    time_ds: h5py.Dataset,
    pose_datasets: list[tuple[str, h5py.Dataset]],
    gripper_ds: h5py.Dataset,
    frame_ds: h5py.Dataset,
) -> int:
    sample_count = int(time_ds.shape[0])
    if sample_count <= 0:
        raise ValueError("hdf5 文件没有可视化样本")
    expected_lengths = {pose_name: int(pose_ds.shape[0]) for pose_name, pose_ds in pose_datasets}
    expected_lengths["gripper_pos"] = int(gripper_ds.shape[0])
    expected_lengths["frame"] = int(frame_ds.shape[0])
    for name, length in expected_lengths.items():
        if length != sample_count:
            raise ValueError(f"{name} 与 time 长度不一致: {length} != {sample_count}")
    for pose_name, pose_ds in pose_datasets:
        if pose_ds.ndim != 2 or pose_ds.shape[1] != 7:
            raise ValueError(f"{pose_name} 形状异常: {pose_ds.shape}")
    if gripper_ds.ndim != 1:
        raise ValueError(f"gripper_pos 形状异常: {gripper_ds.shape}")
    if frame_ds.ndim != 4:
        raise ValueError(f"frame 形状异常: {frame_ds.shape}")
    return sample_count


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


def transform_pose(
    pose: np.ndarray,
    rotation_correction_x_deg: float,
    rotation_correction_y_deg: float,
    rotation_correction_z_deg: float,
) -> np.ndarray:
    pose_array = np.asarray(pose, dtype=np.float64)
    if pose_array.shape != (7,):
        raise ValueError(f"pose 形状异常: {pose_array.shape}")
    if not np.all(np.isfinite(pose_array)):
        return pose_array
    if np.linalg.norm(pose_array[3:]) <= np.finfo(np.float64).eps:
        return pose_array

    aligned_pose = align_tracker_pose(position=pose_array[:3], quaternion=pose_array[3:])
    if aligned_pose is None:
        return pose_array

    aligned_position, aligned_quaternion, _ = aligned_pose
    corrected_quaternion = apply_rotation_correction_x(
        quaternion=aligned_quaternion,
        correction_x_deg=rotation_correction_x_deg,
    )
    corrected_quaternion = apply_rotation_correction_y(
        quaternion=corrected_quaternion,
        correction_y_deg=rotation_correction_y_deg,
    )
    corrected_quaternion = apply_rotation_correction_z(
        quaternion=corrected_quaternion,
        correction_z_deg=rotation_correction_z_deg,
    )
    return np.asarray([*aligned_position, *corrected_quaternion], dtype=np.float64)


def transform_pose_history(
    pose_history: np.ndarray,
    rotation_correction_x_deg: float,
    rotation_correction_y_deg: float,
    rotation_correction_z_deg: float,
) -> np.ndarray:
    if pose_history.size == 0:
        return np.asarray(pose_history, dtype=np.float64)
    return np.asarray(
        [
            transform_pose(
                pose,
                rotation_correction_x_deg=rotation_correction_x_deg,
                rotation_correction_y_deg=rotation_correction_y_deg,
                rotation_correction_z_deg=rotation_correction_z_deg,
            )
            for pose in np.asarray(pose_history, dtype=np.float64)
        ],
        dtype=np.float64,
    )


def get_valid_pose_components(
    pose_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    finite_mask = np.all(np.isfinite(pose_history), axis=1)
    if not np.any(finite_mask):
        return
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


def log_pose_trajectory(
    rr: Any,
    pose_name: str,
    pose_history: np.ndarray,
) -> None:
    pose_components = get_valid_pose_components(pose_history)
    if pose_components is None:
        return
    positions, rotations = pose_components
    log_pose_axes(rr, f"pose/{pose_name}/history", positions, rotations, AXIS_LENGTH, 0.001)
    rr.log(
        f"pose/{pose_name}/current/origin",
        rr.Points3D(positions[-1:], colors=[POSE_COLORS[pose_name]], radii=[0.01]),
    )
    log_pose_axes(rr, f"pose/{pose_name}/current", positions[-1:], rotations[-1:], CURRENT_AXIS_LENGTH, 0.003)


def log_gripper(rr: Any, value: float) -> None:
    if np.isfinite(value):
        rr.log("gripper/position", rr.Scalars(float(value)))


def wait_for_playback(playback_start: float, relative_time: float, playback_rate: float) -> None:
    target_elapsed = max(0.0, relative_time) / playback_rate
    remaining = target_elapsed - (time.perf_counter() - playback_start)
    if remaining > 0:
        time.sleep(remaining)


def visualize_hdf5(
    hdf5_path: Path,
    application_id: str,
    spawn: bool,
    realtime: bool,
    playback_rate: float,
    rotation_correction_x_deg: float,
    rotation_correction_y_deg: float,
    rotation_correction_z_deg: float,
) -> None:
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:
        raise RuntimeError("未安装 rerun-sdk，请先执行 `pip install -r requirements.txt`") from exc

    rr.init(application_id, spawn=spawn)
    rr.send_blueprint(build_blueprint(rrb))

    # 可视化坐标系原点
    world_pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    world_rot = np.array([np.eye(3, dtype=np.float32)])
    log_pose_axes(rr, "pose/world_origin", world_pos, world_rot, 0.1, 0.002)
    rr.log(
        "pose/world_origin/origin",
        rr.Points3D(world_pos, colors=[[255, 255, 255]], radii=[0.005]),
    )

    with h5py.File(hdf5_path, "r") as h5_file:
        time_ds = require_dataset(h5_file, "time")
        pose_datasets = get_pose_datasets(h5_file)
        gripper_ds = require_dataset(h5_file, "gripper_pos")
        frame_ds = require_dataset(h5_file, "frame")
        sample_count = validate_shapes(time_ds, pose_datasets, gripper_ds, frame_ds)

        start_time = float(time_ds[0])
        playback_start = time.perf_counter()
        logger.info("开始可视化: %s", hdf5_path)
        logger.info("样本数: %d", sample_count)
        logger.info("播放模式: %s, 倍率: %.2fx", "实时回放" if realtime else "快速发送", playback_rate)
        logger.info(
            "位姿对齐已启用，旋转校准: x=%.3f°, y=%.3f°, z=%.3f°",
            rotation_correction_x_deg,
            rotation_correction_y_deg,
            rotation_correction_z_deg,
        )

        for index in range(sample_count):
            relative_time = float(time_ds[index]) - start_time
            if realtime:
                wait_for_playback(playback_start, relative_time, playback_rate)
            rr.set_time("frame_idx", sequence=index)
            rr.set_time("time", duration=relative_time)

            for pose_name, pose_ds in pose_datasets:
                pose_history = transform_pose_history(
                    pose_ds[: index + 1],
                    rotation_correction_x_deg=rotation_correction_x_deg,
                    rotation_correction_y_deg=rotation_correction_y_deg,
                    rotation_correction_z_deg=rotation_correction_z_deg,
                )
                log_pose_trajectory(
                    rr,
                    pose_name,
                    pose_history,
                )

            log_gripper(rr, float(gripper_ds[index]))
            rr.log("frame/image", rr.Image(frame_to_rgb(np.asarray(frame_ds[index], dtype=np.uint8))))

        logger.info("Rerun 日志发送完成")


def run() -> int:
    args = parse_args()
    hdf5_path = Path(args.hdf5_path)
    if not hdf5_path.exists():
        logger.error("文件不存在: %s", hdf5_path)
        return 1
    if hdf5_path.suffix.lower() not in {".h5", ".hdf5"}:
        logger.error("仅支持 .h5 或 .hdf5 文件: %s", hdf5_path)
        return 1
    if args.playback_rate <= 0:
        logger.error("--playback-rate 必须大于 0: %s", args.playback_rate)
        return 1
    try:
        visualize_hdf5(
            hdf5_path=hdf5_path,
            application_id=args.application_id,
            spawn=args.spawn,
            realtime=args.realtime,
            playback_rate=args.playback_rate,
            rotation_correction_x_deg=args.rotation_correction_x_deg,
            rotation_correction_y_deg=args.rotation_correction_y_deg,
            rotation_correction_z_deg=args.rotation_correction_z_deg,
        )
    except Exception:
        logger.exception("可视化失败")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
