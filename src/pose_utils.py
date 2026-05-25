from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    归一化四元数，确保其长度为1
    """
    q = np.array(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("四元数不能为零")
    return q / norm


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    q = np.array(q, dtype=float)
    norm_sq = float(np.dot(q, q))
    if norm_sq == 0:
        raise ValueError("四元数不能为零")
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float) / norm_sq


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=float,
    )


def apply_pose_correction(
    position: np.ndarray,
    quaternion: np.ndarray,
    correction_z_deg: float,
    orientation_y_deg: float = 180.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    校准位姿：因为OpenArm的位姿是相对于机器人坐标系的，所以需要进行修正准到目标坐标系
    - 修正z轴：将z轴旋转 correction_z_deg 度
    - 修正y轴：将y轴旋转 orientation_y_deg 度
    """
    correction_rot = R.from_euler("z", correction_z_deg, degrees=True).as_matrix()
    orientation_rot = R.from_euler("y", orientation_y_deg, degrees=True).as_matrix()

    corrected_position = correction_rot @ np.asarray(position, dtype=float)
    corrected_rotation = correction_rot @ R.from_quat(quaternion).as_matrix() @ orientation_rot
    corrected_quaternion = R.from_matrix(corrected_rotation).as_quat()
    return corrected_position, corrected_quaternion


T_WORLD_IN_BASE = np.array(
    [
        [-0.38377469, 0.91041971, 0.15444395],
        [-0.91074653, -0.40079498, 0.0995195],
        [0.15250488, -0.10246623, 0.98297657],
    ],
    dtype=np.float64,
)
T_WORLD_IN_BASE_TRANSLATION = np.array([0.58358624, 0.16126372, 0.02528253], dtype=np.float64)
T_TCP_IN_CAM = np.array(
    [
        [0.06634963, 0.9963019, -0.05459171],
        [-0.73610302, 0.08581082, 0.67140811],
        [0.67360973, -0.00436256, 0.73907435],
    ],
    dtype=np.float64,
)
T_TCP_IN_CAM_TRANSLATION = np.array([0.01668391, 0.041968, 0.23074157], dtype=np.float64)


def build_transform(rotation_matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform


def tracker_pose_to_matrix(position: list[float] | np.ndarray, quaternion: list[float] | np.ndarray) -> np.ndarray | None:
    position_array = np.asarray(position, dtype=np.float64)
    quaternion_array = np.asarray(quaternion, dtype=np.float64)
    if position_array.shape != (3,) or quaternion_array.shape != (4,):
        return None
    if not np.all(np.isfinite(position_array)) or not np.all(np.isfinite(quaternion_array)):
        return None

    quaternion_norm = np.linalg.norm(quaternion_array)
    if quaternion_norm <= np.finfo(np.float64).eps:
        return None

    rotation_matrix = R.from_quat(quaternion_array / quaternion_norm).as_matrix()
    return build_transform(rotation_matrix=rotation_matrix, translation=position_array)


def matrix_to_pose(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return None

    position = transform[:3, 3].copy()
    rotation = R.from_matrix(transform[:3, :3])
    quaternion = rotation.as_quat()
    rotvec = rotation.as_rotvec()
    return position, quaternion, rotvec


def align_tracker_pose(position: list[float] | np.ndarray, quaternion: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    tracker_transform = tracker_pose_to_matrix(position=position, quaternion=quaternion)
    if tracker_transform is None:
        return None

    world_in_base = build_transform(
        rotation_matrix=T_WORLD_IN_BASE,
        translation=T_WORLD_IN_BASE_TRANSLATION,
    )
    tcp_in_cam = build_transform(
        rotation_matrix=T_TCP_IN_CAM,
        translation=T_TCP_IN_CAM_TRANSLATION,
    )
    aligned_transform = world_in_base @ tracker_transform @ tcp_in_cam
    return matrix_to_pose(aligned_transform)


def apply_rotation_correction_y(
    quaternion: list[float] | np.ndarray,
    correction_y_deg: float = -90.0,
) -> np.ndarray:
    """
    应用额外旋转校准 (保持位置不变，仅修改旋转：R' = C * R * C^-1)
    """
    rot_correction = R.from_euler("y", correction_y_deg, degrees=True)
    rot_correction_inv = rot_correction.inv()
    orig_rot = R.from_quat(np.asarray(quaternion, dtype=np.float64))
    corrected_rot = rot_correction * orig_rot * rot_correction_inv
    return corrected_rot.as_quat()