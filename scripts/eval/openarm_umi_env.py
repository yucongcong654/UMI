from __future__ import annotations

import time
from collections import deque
from typing import Any

import cv2
import jax.numpy as jnp
import jaxlie
import numpy as np
import scipy.spatial.transform as st

from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from src.devices.opencv_cam import OpenCVCam
from src.kinematics.openarm_solver import OpenArmIK
from src.pose_utils import (
    T_TCP_IN_CAM,
    T_TCP_IN_CAM_TRANSLATION,
    T_WORLD_IN_BASE,
    T_WORLD_IN_BASE_TRANSLATION,
    build_transform,
)
from umi.common.pose_util import mat_to_pose, pose_to_mat
from umi.common.precise_sleep import precise_wait
from umi.real_world.real_inference_util import get_real_obs_resolution

SOFT_KP_SCALE = 0.65
SOFT_KD_SCALE = 0.8
GRIPPER_CLOSE_SNAP_RATIO = 0.1

WORLD_IN_BASE = build_transform(T_WORLD_IN_BASE, T_WORLD_IN_BASE_TRANSLATION)
TRACKER_TO_EE = build_transform(T_TCP_IN_CAM, T_TCP_IN_CAM_TRANSLATION)
BASE_IN_WORLD = np.linalg.inv(WORLD_IN_BASE)
EE_IN_TRACKER = np.linalg.inv(TRACKER_TO_EE)


def parse_camera_device(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def stack_history(items: deque[np.ndarray | float], horizon: int) -> np.ndarray:
    data = list(items)
    if not data:
        raise RuntimeError("Observation history is empty.")
    if len(data) < horizon:
        data = [data[0]] * (horizon - len(data)) + data
    else:
        data = data[-horizon:]
    first = data[0]
    if isinstance(first, np.ndarray):
        return np.stack(data, axis=0)
    return np.asarray(data, dtype=np.float64)


def normalize_env_action_sequence(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    if action.ndim == 1:
        if action.shape[0] != 7:
            raise ValueError(f"Expected a 7D env action, got shape {action.shape}")
        return action[None]
    if action.ndim != 2 or action.shape[-1] != 7:
        raise ValueError(f"Expected an action sequence with shape [T, 7], got {action.shape}")
    return action


def pose_to_se3(pose: np.ndarray) -> jaxlie.SE3:
    rotvec = jnp.asarray(pose[3:], dtype=jnp.float32)
    trans = jnp.asarray(pose[:3], dtype=jnp.float32)
    if hasattr(jaxlie.SO3, "from_matrix"):
        rot_m = st.Rotation.from_rotvec(np.asarray(pose[3:], dtype=np.float64)).as_matrix()
        so3 = jaxlie.SO3.from_matrix(jnp.asarray(rot_m, dtype=jnp.float32))
    else:
        so3 = jaxlie.SO3.exp(rotvec)

    if hasattr(jaxlie.SE3, "from_rotation_and_translation"):
        return jaxlie.SE3.from_rotation_and_translation(so3, trans)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(so3.as_matrix(), dtype=np.float32)
    transform[:3, 3] = np.asarray(trans, dtype=np.float32)
    return jaxlie.SE3.from_matrix(jnp.asarray(transform))


def tracker_pose_mat_to_robot_ee_mat(tracker_pose_mat: np.ndarray) -> np.ndarray:
    return WORLD_IN_BASE @ tracker_pose_mat @ TRACKER_TO_EE


def robot_ee_pose_mat_to_tracker_mat(robot_ee_pose_mat: np.ndarray) -> np.ndarray:
    return BASE_IN_WORLD @ robot_ee_pose_mat @ EE_IN_TRACKER


def _interpolate_robot_action(
    target_action: dict[str, float],
    previous_action: dict[str, float] | None,
    alpha: float,
) -> dict[str, float]:
    if previous_action is None or alpha >= 1.0:
        return {key: float(value) for key, value in target_action.items()}

    interpolated_action: dict[str, float] = {}
    for key, target_value in target_action.items():
        prev_value = previous_action.get(key, float(target_value))
        interpolated_action[key] = float(prev_value) + (float(target_value) - float(prev_value)) * alpha
    return interpolated_action


class OpenArmUmiEnv:
    """Expose tracker-space observations while executing aligned end-effector targets."""

    def __init__(
        self,
        robot: OpenArmFollower,
        camera: OpenCVCam,
        ik_solver: OpenArmIK,
        side: str,
        shape_meta: dict[str, Any],
        gripper_open_deg: float,
        gripper_closed_deg: float,
        policy_gripper_open_value: float,
        policy_gripper_closed_value: float,
        action_interp_alpha: float,
        tracker_rotation_correction_y_deg: float = 0.0,
        visualize_camera: bool = False,
        preview_window_name: str | None = None,
    ) -> None:
        self.robot = robot
        self.camera = camera
        self.ik_solver = ik_solver
        self.side = side
        self.shape_meta = shape_meta
        self.gripper_open_deg = gripper_open_deg
        self.gripper_closed_deg = gripper_closed_deg
        self.policy_gripper_open_value = policy_gripper_open_value
        self.policy_gripper_closed_value = policy_gripper_closed_value
        self.action_interp_alpha = float(np.clip(action_interp_alpha, 0.0, 1.0))
        self.tracker_rotation_correction_y_deg = float(tracker_rotation_correction_y_deg)
        self.visualize_camera = bool(visualize_camera)
        self.preview_window_name = preview_window_name or f"OpenArmUmiCamera[{side}]"
        self._preview_available = self.visualize_camera
        self._preview_error_logged = False
        self.previous_motor_action: dict[str, float] | None = None
        self.custom_kp, self.custom_kd = self._build_soft_motor_gains()

        obs_shape_meta = shape_meta["obs"]
        self.rgb_keys = [k for k, v in obs_shape_meta.items() if v.get("type", "low_dim") == "rgb"]
        if not self.rgb_keys:
            raise ValueError("The checkpoint does not contain any RGB observation keys.")
        self.primary_rgb_key = self.rgb_keys[0]
        self.key_horizon = {k: int(v.get("horizon", 1)) for k, v in obs_shape_meta.items()}
        self.max_horizon = max(self.key_horizon.values())
        self.obs_res = get_real_obs_resolution(shape_meta)
        self.image_transform = None

        self.control_joint_names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        self.control_joint_indices = [self.ik_solver._qidx[name] for name in self.control_joint_names]

        self.history: dict[str, deque[np.ndarray | float]] = {
            "timestamp": deque(maxlen=self.max_horizon),
            "image": deque(maxlen=self.max_horizon),
            "robot0_eef_pos": deque(maxlen=self.max_horizon),
            "robot0_eef_rot_axis_angle": deque(maxlen=self.max_horizon),
            "robot0_gripper_width": deque(maxlen=self.max_horizon),
        }

    def _build_soft_motor_gains(self) -> tuple[dict[str, float], dict[str, float]]:
        motor_names = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]
        config_kp = list(self.robot.config.position_kp)
        config_kd = list(self.robot.config.position_kd)
        custom_kp = {
            motor_name: float(kp) * SOFT_KP_SCALE for motor_name, kp in zip(motor_names, config_kp, strict=True)
        }
        custom_kd = {
            motor_name: float(kd) * SOFT_KD_SCALE for motor_name, kd in zip(motor_names, config_kd, strict=True)
        }
        return custom_kp, custom_kd

    def _resize_rgb(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"Unexpected camera frame shape: {image.shape}")
        image = image[..., :3]

        # Match the training-time preprocessing in umi_hdf5_dataset.py.
        out = cv2.resize(image, self.obs_res, interpolation=cv2.INTER_AREA)

        if out.dtype == np.uint8:
            out = out.astype(np.float32) / 255.0
        else:
            out = out.astype(np.float32)
        return np.ascontiguousarray(out)

    def _show_camera_preview(self, frame: np.ndarray) -> None:
        if not self._preview_available:
            return

        try:
            cv2.imshow(self.preview_window_name, frame)
            cv2.waitKey(1)
        except cv2.error as exc:
            if not self._preview_error_logged:
                print(f"Camera preview disabled: {exc}")
                self._preview_error_logged = True
            self._preview_available = False
            self._close_camera_preview()

    def _close_camera_preview(self) -> None:
        if not self.visualize_camera:
            return
        try:
            cv2.destroyWindow(self.preview_window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass

    def _gripper_pos_to_policy_value(self, gripper_pos: float) -> float:
        lo = min(self.gripper_open_deg, self.gripper_closed_deg)
        hi = max(self.gripper_open_deg, self.gripper_closed_deg)
        gripper_pos = float(np.clip(gripper_pos, lo, hi))
        gripper_pos = self._snap_gripper_pos_to_closed(gripper_pos)
        alpha = (gripper_pos - self.gripper_closed_deg) / (
            self.gripper_open_deg - self.gripper_closed_deg + 1e-8
        )
        return float(
            self.policy_gripper_closed_value
            + np.clip(alpha, 0.0, 1.0)
            * (self.policy_gripper_open_value - self.policy_gripper_closed_value)
        )

    def _snap_gripper_pos_to_closed(self, gripper_pos: float) -> float:
        gripper_range = abs(self.gripper_open_deg - self.gripper_closed_deg)
        if gripper_range <= 1e-8:
            return float(self.gripper_closed_deg)
        if abs(gripper_pos - self.gripper_closed_deg) <= gripper_range * GRIPPER_CLOSE_SNAP_RATIO:
            return float(self.gripper_closed_deg)
        return float(gripper_pos)

    def _policy_gripper_to_pos(self, policy_gripper_value: float) -> float:
        lo = min(self.policy_gripper_open_value, self.policy_gripper_closed_value)
        hi = max(self.policy_gripper_open_value, self.policy_gripper_closed_value)
        value = float(np.clip(policy_gripper_value, lo, hi))
        alpha = (value - self.policy_gripper_closed_value) / (
            self.policy_gripper_open_value - self.policy_gripper_closed_value + 1e-8
        )
        gripper_pos = float(
            self.gripper_closed_deg + np.clip(alpha, 0.0, 1.0) * (self.gripper_open_deg - self.gripper_closed_deg)
        )
        return self._snap_gripper_pos_to_closed(gripper_pos)

    def _observation_to_full_q(self, robot_obs: dict[str, Any]) -> np.ndarray:
        q = np.asarray(self.ik_solver.get_default_config(), dtype=np.float64)
        for idx, q_idx in enumerate(self.control_joint_indices, start=1):
            q[q_idx] = np.deg2rad(float(robot_obs[f"joint_{idx}.pos"]))
        return q

    def _apply_tracker_rotation_correction(self, pose: np.ndarray, inverse: bool = False) -> np.ndarray:
        corrected_pose = np.asarray(pose, dtype=np.float64).copy()
        if np.isclose(self.tracker_rotation_correction_y_deg, 0.0):
            return corrected_pose

        rot_correction = st.Rotation.from_euler("y", self.tracker_rotation_correction_y_deg, degrees=True)
        if inverse:
            rot_correction = rot_correction.inv()

        C_mat = np.eye(4, dtype=np.float64)
        C_mat[:3, :3] = rot_correction.as_matrix()

        T_orig = pose_to_mat(corrected_pose)
        T_corrected = C_mat @ T_orig @ np.linalg.inv(C_mat)

        return mat_to_pose(T_corrected)

    def _read_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
        robot_obs = self.robot.get_observation()
        image, image_time = self.camera.read_once(copy_frame=True, return_time=True)
        if image is None:
            raise RuntimeError("Failed to read a frame from OpenCVCam.")
        self._show_camera_preview(np.asarray(image))
        q_full = self._observation_to_full_q(robot_obs)
        ee_pose = self.ik_solver.forward_kinematics(jnp.asarray(q_full))[self.side]

        pos = np.asarray(ee_pose.translation(), dtype=np.float64)
        rot_m = np.asarray(ee_pose.rotation().as_matrix(), dtype=np.float64)
        rotvec = st.Rotation.from_matrix(rot_m).as_rotvec()
        robot_ee_pose = np.concatenate([pos, rotvec], axis=-1)

        # Apply rotation correction on the aligned pose first, then convert back to tracker pose
        rotated_ee_pose = self._apply_tracker_rotation_correction(robot_ee_pose, inverse=False)
        tracker_pose = mat_to_pose(
            robot_ee_pose_mat_to_tracker_mat(pose_to_mat(rotated_ee_pose))
        )

        image = self._resize_rgb(np.asarray(image))
        gripper_policy_value = self._gripper_pos_to_policy_value(float(robot_obs["gripper.pos"]))
        timestamp = float(image_time) if image_time > 0 else time.time()
        return (
            image,
            tracker_pose[:3],
            tracker_pose[3:],
            gripper_policy_value,
            timestamp,
            q_full,
        )

    def get_obs(self) -> dict[str, np.ndarray]:
        image, pos, rotvec, gripper_policy_value, timestamp, _ = self._read_state()
        self.history["timestamp"].append(timestamp)
        self.history["image"].append(image)
        self.history["robot0_eef_pos"].append(pos.astype(np.float32))
        self.history["robot0_eef_rot_axis_angle"].append(rotvec.astype(np.float32))
        self.history["robot0_gripper_width"].append(np.asarray([gripper_policy_value], dtype=np.float32))

        obs: dict[str, np.ndarray] = {
            "timestamp": stack_history(self.history["timestamp"], self.key_horizon[self.primary_rgb_key]),
            "robot0_eef_pos": stack_history(
                self.history["robot0_eef_pos"], self.key_horizon["robot0_eef_pos"]
            ).astype(np.float32),
            "robot0_eef_rot_axis_angle": stack_history(
                self.history["robot0_eef_rot_axis_angle"], self.key_horizon["robot0_eef_rot_axis_angle"]
            ).astype(np.float32),
            "robot0_gripper_width": stack_history(
                self.history["robot0_gripper_width"], self.key_horizon["robot0_gripper_width"]
            ).astype(np.float32),
        }
        image_history = stack_history(self.history["image"], self.key_horizon[self.primary_rgb_key]).astype(np.float32)
        for key in self.rgb_keys:
            obs[key] = image_history
        return obs

    def exec_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"Expected a 7D env action, got shape {action.shape}")

        _, _, _, _, _, q_current = self._read_state()
        
        # Convert tracker pose to aligned robot ee pose first, then apply inverse rotation correction
        robot_ee_target_pose_rot = mat_to_pose(
            tracker_pose_mat_to_robot_ee_mat(pose_to_mat(np.copy(action[:6])))
        )
        robot_ee_target_pose = self._apply_tracker_rotation_correction(robot_ee_target_pose_rot, inverse=True)
        
        target_se3 = pose_to_se3(robot_ee_target_pose)
        if self.side == "left":
            q_next = self.ik_solver.solve(target_L=target_se3, target_R=None, q_current=jnp.asarray(q_current))
        else:
            q_next = self.ik_solver.solve(target_L=None, target_R=target_se3, q_current=jnp.asarray(q_current))

        q_next = np.asarray(q_next, dtype=np.float64)
        motor_action = {
            f"joint_{i}.pos": float(np.rad2deg(q_next[q_idx]))
            for i, q_idx in enumerate(self.control_joint_indices, start=1)
        }
        motor_action = _interpolate_robot_action(
            target_action=motor_action,
            previous_action=self.previous_motor_action,
            alpha=self.action_interp_alpha,
        )
        motor_action["gripper.pos"] = self._policy_gripper_to_pos(float(action[6]))
        self.robot.send_action(motor_action, custom_kp=self.custom_kp, custom_kd=self.custom_kd)
        self.previous_motor_action = motor_action

    def exec_actions(self, actions: np.ndarray, timestamps: np.ndarray) -> int:
        action_sequence = normalize_env_action_sequence(actions)
        timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(action_sequence) != len(timestamps):
            raise ValueError(
                f"Action count {len(action_sequence)} does not match timestamp count {len(timestamps)}"
            )

        executed = 0
        for action, target_time in zip(action_sequence, timestamps):
            if target_time > time.time():
                precise_wait(target_time, time_func=time.time)
            self.exec_action(action)
            executed += 1
        return executed

    def _send_joint_target(
        self, joint_targets_deg: dict[str, float], duration_s: float = 2.0, command_hz: float = 20.0
    ) -> None:
        """Send the same joint target for a short duration to make the motion stable."""
        n_steps = max(1, int(duration_s * command_hz))
        dt = duration_s / n_steps
        target_action = {key: float(value) for key, value in joint_targets_deg.items()}
        commanded_action = self.previous_motor_action
        for _ in range(n_steps):
            commanded_action = _interpolate_robot_action(
                target_action=target_action,
                previous_action=commanded_action,
                alpha=self.action_interp_alpha,
            )
            self.robot.send_action(commanded_action, custom_kp=self.custom_kp, custom_kd=self.custom_kd)
            precise_wait(time.monotonic() + dt)
        self.previous_motor_action = commanded_action

    def move_to_default_pose(self, duration_s: float = 2.0, command_hz: float = 20.0) -> None:
        """Move the controlled arm to the default joint pose defined in `OpenArmIK`."""
        q_default = np.asarray(self.ik_solver.get_default_config(), dtype=np.float64)
        motor_action = {
            f"joint_{i}.pos": float(np.rad2deg(q_default[q_idx]))
            for i, q_idx in enumerate(self.control_joint_indices, start=1)
        }
        motor_action["gripper.pos"] = 0.0
        self._send_joint_target(motor_action, duration_s=duration_s, command_hz=command_hz)

    def move_to_zero_pose(self, duration_s: float = 2.0, command_hz: float = 20.0) -> None:
        """Move the controlled arm to an all-zero joint configuration before shutdown."""
        motor_action = {f"joint_{i}.pos": 0.0 for i in range(1, 8)}
        motor_action["gripper.pos"] = 0.0
        self._send_joint_target(motor_action, duration_s=duration_s, command_hz=command_hz)

    def close(self) -> None:
        self._close_camera_preview()


def make_robot(
    robot_id: str,
    side: str,
    port: str,
    can_interface: str,
    max_relative_target: float,
) -> OpenArmFollower:
    config = OpenArmFollowerConfig(
        id=robot_id,
        port=port,
        side=side,
        can_interface=can_interface,
        max_relative_target=max_relative_target,
    )
    return OpenArmFollower(config)


def make_camera(
    camera_device: str,
    camera_width: int | None,
    camera_height: int | None,
    camera_fps: int,
) -> OpenCVCam:
    return OpenCVCam(
        device_id=parse_camera_device(camera_device),
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fps=camera_fps,
    )
