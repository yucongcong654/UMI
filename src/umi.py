from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.devices.gripper_serial import Esp32gripperReader
from src.devices.opencv_cam import OpenCVCam
from src.devices.tracker import ViveTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("umi.record_pose")


class UMI:
    def __init__(
        self,
        tracker_config: str,
        camera_device_id: int = 0,
        gripper_serial_port: str = "/dev/ttyACM0",
        tracker_device_name: str = None,
        rotation_correction_y_deg: float = -90.0,
        calibrate_pose: bool = True,
    ) -> None:
        # Initialize devices
        self.tracker = ViveTracker(config_path=tracker_config)
        self.tracker_device_name = tracker_device_name
        self.cam = OpenCVCam(device_id=camera_device_id)
        self.gripper = Esp32gripperReader(port=gripper_serial_port)
        self.rotation_correction_y_deg = rotation_correction_y_deg
        self.calibrate_pose = calibrate_pose

    @property
    def is_connected(self) -> bool:
        return self.tracker.is_connected and self.cam.is_connected and self.gripper.is_connected

    def connect(self) -> bool:
        if self.is_connected:
            return True
        if not self.tracker.connect():
            return False
        if not self.cam.connect():
            self.tracker.disconnect()
            return False
        if not self.gripper.connect():
            self.gripper.disconnect()
            self.cam.disconnect()
            self.tracker.disconnect()
            return False
        return True

    def disconnect(self) -> None:
        self.cam.disconnect()
        self.tracker.disconnect()
        self.gripper.disconnect()

    def read_once(self, copy_frame: bool = True) -> Dict[str, Any]:
        """
        读取一次 UMI 数据。
        1. 从相机读取一帧图像。
        2. 从跟踪器读取 UMI 姿态。
        3. 从 gripper 读取位置。
        4. 计算观测时间。
        5. 返回包含时间、姿态、图像和 gripper 位置的字典。
        """
        frame, camera_time = self.cam.read_once(copy_frame=copy_frame, return_time=True)
        pose = self.tracker.read_once(device_name=self.tracker_device_name) if self.tracker_device_name else None
        gripper_pos = self.gripper.read_once()
        observation_time = float(camera_time) if camera_time > 0 else time.time()
        return {
            "time": observation_time,
            "pose": pose,
            "frame": frame,
            "gripper_pos": gripper_pos,
        }

    def record_episode(
        self,
        file_path: str | Path,
        sample_hz: Optional[float] = None,
        overwrite: bool = False,
        wait_for_ready_seconds: float = 5.0,
        stop_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """
        记录 UMI 数据。
        1. 连接 UMI 设备。
        2. 等待 UMI 设备就绪。
        3. 记录 UMI 数据，直到停止事件触发。
        4. 关闭 UMI 设备。
        5. 返回包含记录时间、结束时间、样本数和记录错误的字典。
        """
        output_path = Path(file_path)
        if output_path.suffix.lower() not in {".h5", ".hdf5"}:
            output_path = output_path.with_suffix(".hdf5")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"输出文件已存在: {output_path}")

        if not self.connect():
            raise RuntimeError("连接 UMI 设备失败")

        first_observation = self._wait_for_observation(timeout_seconds=wait_for_ready_seconds)
        if first_observation is None:
            raise RuntimeError("未能读取到完整观测")

        sample_interval = 1.0 / sample_hz if sample_hz and sample_hz > 0 else 0.0
        wall_start_time = time.time()
        start_time: Optional[float] = None
        end_time: Optional[float] = None
        sample_count = 0
        last_observation_time = -1.0
        record_error: Optional[str] = None

        try:
            with h5py.File(output_path, "w") as h5_file:
                datasets = self._initialize_hdf5_file(h5_file=h5_file, frame=first_observation["frame"], sample_hz=sample_hz)
                observation = first_observation
                while True:
                    loop_start = time.time()
                    observation_time = float(observation["time"])
                    if observation["frame"] is not None and observation_time > last_observation_time:
                        self._append_sample(
                            datasets=datasets,
                            index=sample_count,
                            observation_time=observation_time,
                            pose=self._pose_to_array(observation["pose"]),
                            frame=observation["frame"],
                            gripper_pos=observation["gripper_pos"],
                        )
                        if start_time is None:
                            start_time = observation_time
                        end_time = observation_time
                        sample_count += 1
                        last_observation_time = observation_time

                    if stop_event is not None:
                        if stop_event.is_set():
                            break
                        if sample_interval > 0:
                            elapsed = time.time() - loop_start
                            wait_seconds = max(0.0, sample_interval - elapsed)
                            if stop_event.wait(timeout=wait_seconds):
                                break
                        elif stop_event.wait(timeout=0.001):
                            break
                    else:
                        if sample_interval > 0:
                            elapsed = time.time() - loop_start
                            wait_seconds = max(0.0, sample_interval - elapsed)
                            if wait_seconds > 0:
                                time.sleep(wait_seconds)
                        else:
                            time.sleep(0.001)

                    observation = self.read_once()
        except Exception as exc:
            logger.exception("录制 UMI 数据时发生异常")
            record_error = str(exc)
            raise
        finally:
            wall_end_time = time.time()
            record_start_time = start_time if start_time is not None else wall_start_time
            record_end_time = end_time if end_time is not None else wall_end_time
            self._finalize_hdf5_file(
                file_path=output_path,
                start_time=record_start_time,
                end_time=record_end_time,
                sample_hz=sample_hz,
                sample_count=sample_count,
                record_error=record_error,
                wall_start_time=wall_start_time,
                wall_end_time=wall_end_time,
            )

        return {
            "hdf5_path": str(output_path),
            "start_time": record_start_time,
            "end_time": record_end_time,
            "duration": max(0.0, record_end_time - record_start_time),
            "wall_start_time": wall_start_time,
            "wall_end_time": wall_end_time,
            "wall_duration": max(0.0, wall_end_time - wall_start_time),
            "sample_hz": sample_hz if sample_hz is not None else float(self.cam.camera_fps),
            "sample_count": sample_count,
            "tracker_device": self.tracker_device_name,
            "record_error": record_error,
        }

    def _wait_for_observation(self, timeout_seconds: float) -> Optional[Dict[str, Any]]:
        """
        等待 UMI 数据采集到完整观测，超时后返回 None。
        """
        deadline = time.time() + max(timeout_seconds, 0.0)
        while time.time() <= deadline:
            observation = self.read_once(copy_frame=True)
            if observation["frame"] is not None:
                return observation
            time.sleep(0.02)
        return None

    def _initialize_hdf5_file(
        self,
        h5_file: h5py.File,
        frame: np.ndarray,
        sample_hz: Optional[float],
    ) -> Dict[str, h5py.Dataset]:
        """
        初始化 HDF5 文件，创建数据集。
        """
        height, width, channels = frame.shape
        h5_file.attrs["tracker_device"] = self.tracker_device_name if self.tracker_device_name else ""
        h5_file.attrs["camera_width"] = int(width)
        h5_file.attrs["camera_height"] = int(height)
        h5_file.attrs["camera_channels"] = int(channels)
        h5_file.attrs["sample_hz"] = float(sample_hz if sample_hz is not None else self.cam.camera_fps)
        return {
            "time": h5_file.create_dataset(
                "time",
                shape=(0,),
                maxshape=(None,),
                chunks=True,
                dtype=np.float64,
            ),
            "pose": h5_file.create_dataset(
                "pose",
                shape=(0, 7),
                maxshape=(None, 7),
                chunks=True,
                dtype=np.float64,
            ),
            "gripper_pos": h5_file.create_dataset(
                "gripper_pos",
                shape=(0,),
                maxshape=(None,),
                chunks=True,
                dtype=np.float64,
            ),
            "frame": h5_file.create_dataset(
                "frame",
                shape=(0, height, width, channels),
                maxshape=(None, height, width, channels),
                chunks=(1, height, width, channels),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=4,
            ),
        }

    def _append_sample(
        self,
        datasets: Dict[str, h5py.Dataset],
        index: int,
        observation_time: float,
        pose: np.ndarray,
        frame: np.ndarray,
        gripper_pos: Optional[float],
    ) -> None:
        """
        将 UMI 数据追加到 HDF5 文件中。
        """
        for dataset in datasets.values():
            dataset.resize(index + 1, axis=0)
        datasets["time"][index] = observation_time
        datasets["pose"][index] = pose
        datasets["gripper_pos"][index] = np.nan if gripper_pos is None else float(gripper_pos)
        datasets["frame"][index] = frame

    def _pose_to_array(self, pose: Any) -> np.ndarray:
        """
        将 UMI 姿态转换为 numpy 数组。
        """
        if pose is None:
            return np.full((7,), np.nan, dtype=np.float64)

        return np.asarray([*pose.position, *pose.rotation], dtype=np.float64)

    def _finalize_hdf5_file(
        self,
        file_path: Path,
        start_time: float,
        end_time: float,
        sample_hz: Optional[float],
        sample_count: int,
        record_error: Optional[str],
        wall_start_time: float,
        wall_end_time: float,
    ) -> None:
        if not file_path.exists():
            return
        with h5py.File(file_path, "a") as h5_file:
            h5_file.attrs["start_time"] = float(start_time)
            h5_file.attrs["end_time"] = float(end_time)
            h5_file.attrs["duration"] = float(max(0.0, end_time - start_time))
            h5_file.attrs["wall_start_time"] = float(wall_start_time)
            h5_file.attrs["wall_end_time"] = float(wall_end_time)
            h5_file.attrs["wall_duration"] = float(max(0.0, wall_end_time - wall_start_time))
            h5_file.attrs["sample_hz"] = float(sample_hz if sample_hz is not None else self.cam.camera_fps)
            h5_file.attrs["sample_count"] = int(sample_count)
            h5_file.attrs["record_error"] = record_error if record_error else ""

    def __del__(self) -> None:
        try:
            self.disconnect()
        except Exception:
            pass


class BimanualUMI:
    def __init__(
        self,
        tracker_config: str,
        left_camera_device_id: int = 0,
        right_camera_device_id: int = 1,
        left_gripper_serial_port: str = "/dev/ttyACM0",
        right_gripper_serial_port: str = "/dev/ttyACM1",
        left_tracker_device_name: str = None,
        right_tracker_device_name: str = None,
        rotation_correction_y_deg: float = -90.0,
        calibrate_pose: bool = True,
    ) -> None:
        self.tracker = ViveTracker(config_path=tracker_config)
        self.left_tracker_device_name = left_tracker_device_name
        self.left_cam = OpenCVCam(device_id=left_camera_device_id)
        self.left_gripper = Esp32gripperReader(port=left_gripper_serial_port)

        self.right_tracker_device_name = right_tracker_device_name
        self.right_cam = OpenCVCam(device_id=right_camera_device_id)
        self.right_gripper = Esp32gripperReader(port=right_gripper_serial_port)
        self.rotation_correction_y_deg = rotation_correction_y_deg
        self.calibrate_pose = calibrate_pose

    @property
    def is_connected(self) -> bool:
        return self.tracker.is_connected and self._side_is_connected("left") and self._side_is_connected("right")

    def connect(self) -> bool:
        if self.is_connected:
            return True
        if not self.tracker.connect():
            return False
        if not self._connect_side("left"):
            self.disconnect()
            return False
        if not self._connect_side("right"):
            self.disconnect()
            return False
        return True

    def disconnect(self) -> None:
        self._disconnect_side("left")
        self._disconnect_side("right")
        self.tracker.disconnect()

    def read_once(self, copy_frame: bool = True) -> Dict[str, Any]:
        observation: Dict[str, Any] = {}
        for prefix in ("left", "right"):
            side_observation = self._read_side_once(prefix=prefix, copy_frame=copy_frame)
            for key, value in side_observation.items():
                observation[f"{prefix}_{key}"] = value
        return observation

    def record_episode(
        self,
        file_path: str | Path,
        sample_hz: Optional[float] = None,
        overwrite: bool = False,
        wait_for_ready_seconds: float = 5.0,
        stop_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        output_path = Path(file_path)
        if output_path.suffix.lower() not in {".h5", ".hdf5"}:
            output_path = output_path.with_suffix(".hdf5")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"输出文件已存在: {output_path}")

        if not self.connect():
            raise RuntimeError("连接双臂 UMI 设备失败")

        first_observation = self._wait_for_observation(timeout_seconds=wait_for_ready_seconds)
        if first_observation is None:
            raise RuntimeError("未能读取到左右手完整观测")

        sample_interval = 1.0 / sample_hz if sample_hz and sample_hz > 0 else 0.0
        wall_start_time = time.time()
        start_time: Optional[float] = None
        end_time: Optional[float] = None
        sample_count = 0
        last_left_time = -1.0
        last_right_time = -1.0
        record_error: Optional[str] = None

        try:
            with h5py.File(output_path, "w") as h5_file:
                datasets = self._initialize_hdf5_file(
                    h5_file=h5_file,
                    left_frame=first_observation["left_frame"],
                    right_frame=first_observation["right_frame"],
                    sample_hz=sample_hz,
                )
                observation = first_observation
                while True:
                    loop_start = time.time()
                    left_time = float(observation["left_time"])
                    right_time = float(observation["right_time"])
                    has_frames = observation["left_frame"] is not None and observation["right_frame"] is not None
                    has_new_sample = left_time > last_left_time or right_time > last_right_time
                    if has_frames and has_new_sample:
                        self._append_sample(
                            datasets=datasets,
                            index=sample_count,
                            observation=observation,
                        )
                        if start_time is None:
                            start_time = min(left_time, right_time)
                        end_time = max(left_time, right_time)
                        sample_count += 1
                        last_left_time = left_time
                        last_right_time = right_time

                    if stop_event is not None:
                        if stop_event.is_set():
                            break
                        if sample_interval > 0:
                            elapsed = time.time() - loop_start
                            wait_seconds = max(0.0, sample_interval - elapsed)
                            if stop_event.wait(timeout=wait_seconds):
                                break
                        elif stop_event.wait(timeout=0.001):
                            break
                    else:
                        if sample_interval > 0:
                            elapsed = time.time() - loop_start
                            wait_seconds = max(0.0, sample_interval - elapsed)
                            if wait_seconds > 0:
                                time.sleep(wait_seconds)
                        else:
                            time.sleep(0.001)

                    observation = self.read_once()
        except Exception as exc:
            logger.exception("录制双臂 UMI 数据时发生异常")
            record_error = str(exc)
            raise
        finally:
            wall_end_time = time.time()
            record_start_time = start_time if start_time is not None else wall_start_time
            record_end_time = end_time if end_time is not None else wall_end_time
            self._finalize_hdf5_file(
                file_path=output_path,
                start_time=record_start_time,
                end_time=record_end_time,
                sample_hz=sample_hz,
                sample_count=sample_count,
                record_error=record_error,
                wall_start_time=wall_start_time,
                wall_end_time=wall_end_time,
            )

        return {
            "hdf5_path": str(output_path),
            "start_time": record_start_time,
            "end_time": record_end_time,
            "duration": max(0.0, record_end_time - record_start_time),
            "wall_start_time": wall_start_time,
            "wall_end_time": wall_end_time,
            "wall_duration": max(0.0, wall_end_time - wall_start_time),
            "sample_hz": sample_hz if sample_hz is not None else self._default_sample_hz(),
            "sample_count": sample_count,
            "left_tracker_device": self.left_tracker_device_name,
            "right_tracker_device": self.right_tracker_device_name,
            "record_error": record_error,
        }

    def _wait_for_observation(self, timeout_seconds: float) -> Optional[Dict[str, Any]]:
        deadline = time.time() + max(timeout_seconds, 0.0)
        while time.time() <= deadline:
            observation = self.read_once(copy_frame=True)
            if observation["left_frame"] is not None and observation["right_frame"] is not None:
                return observation
            time.sleep(0.02)
        return None

    def _initialize_hdf5_file(
        self,
        h5_file: h5py.File,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        sample_hz: Optional[float],
    ) -> Dict[str, h5py.Dataset]:
        self._set_side_attributes(h5_file=h5_file, prefix="left", frame=left_frame)
        self._set_side_attributes(h5_file=h5_file, prefix="right", frame=right_frame)
        h5_file.attrs["sample_hz"] = float(sample_hz if sample_hz is not None else self._default_sample_hz())
        datasets: Dict[str, h5py.Dataset] = {}
        datasets.update(self._create_side_datasets(h5_file=h5_file, prefix="left", frame=left_frame))
        datasets.update(self._create_side_datasets(h5_file=h5_file, prefix="right", frame=right_frame))
        return datasets

    def _set_side_attributes(self, h5_file: h5py.File, prefix: str, frame: np.ndarray) -> None:
        height, width, channels = frame.shape
        tracker_device_name = getattr(self, f"{prefix}_tracker_device_name")
        h5_file.attrs[f"{prefix}_tracker_device"] = tracker_device_name if tracker_device_name else ""
        h5_file.attrs[f"{prefix}_camera_width"] = int(width)
        h5_file.attrs[f"{prefix}_camera_height"] = int(height)
        h5_file.attrs[f"{prefix}_camera_channels"] = int(channels)

    def _create_side_datasets(self, h5_file: h5py.File, prefix: str, frame: np.ndarray) -> Dict[str, h5py.Dataset]:
        height, width, channels = frame.shape
        return {
            f"{prefix}_time": h5_file.create_dataset(
                f"{prefix}_time",
                shape=(0,),
                maxshape=(None,),
                chunks=True,
                dtype=np.float64,
            ),
            f"{prefix}_pose": h5_file.create_dataset(
                f"{prefix}_pose",
                shape=(0, 7),
                maxshape=(None, 7),
                chunks=True,
                dtype=np.float64,
            ),
            f"{prefix}_gripper_pos": h5_file.create_dataset(
                f"{prefix}_gripper_pos",
                shape=(0,),
                maxshape=(None,),
                chunks=True,
                dtype=np.float64,
            ),
            f"{prefix}_frame": h5_file.create_dataset(
                f"{prefix}_frame",
                shape=(0, height, width, channels),
                maxshape=(None, height, width, channels),
                chunks=(1, height, width, channels),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=4,
            ),
        }

    def _append_sample(
        self,
        datasets: Dict[str, h5py.Dataset],
        index: int,
        observation: Dict[str, Any],
    ) -> None:
        for dataset in datasets.values():
            dataset.resize(index + 1, axis=0)
        for prefix in ("left", "right"):
            datasets[f"{prefix}_time"][index] = float(observation[f"{prefix}_time"])
            datasets[f"{prefix}_pose"][index] = self._pose_to_array(observation[f"{prefix}_pose"])
            gripper_pos = observation[f"{prefix}_gripper_pos"]
            datasets[f"{prefix}_gripper_pos"][index] = np.nan if gripper_pos is None else float(gripper_pos)
            datasets[f"{prefix}_frame"][index] = observation[f"{prefix}_frame"]

    def _finalize_hdf5_file(
        self,
        file_path: Path,
        start_time: float,
        end_time: float,
        sample_hz: Optional[float],
        sample_count: int,
        record_error: Optional[str],
        wall_start_time: float,
        wall_end_time: float,
    ) -> None:
        if not file_path.exists():
            return
        with h5py.File(file_path, "a") as h5_file:
            h5_file.attrs["start_time"] = float(start_time)
            h5_file.attrs["end_time"] = float(end_time)
            h5_file.attrs["duration"] = float(max(0.0, end_time - start_time))
            h5_file.attrs["wall_start_time"] = float(wall_start_time)
            h5_file.attrs["wall_end_time"] = float(wall_end_time)
            h5_file.attrs["wall_duration"] = float(max(0.0, wall_end_time - wall_start_time))
            h5_file.attrs["sample_hz"] = float(sample_hz if sample_hz is not None else self._default_sample_hz())
            h5_file.attrs["sample_count"] = int(sample_count)
            h5_file.attrs["record_error"] = record_error if record_error else ""

    def _side_is_connected(self, prefix: str) -> bool:
        cam = getattr(self, f"{prefix}_cam")
        gripper = getattr(self, f"{prefix}_gripper")
        return cam.is_connected and gripper.is_connected

    def _connect_side(self, prefix: str) -> bool:
        cam = getattr(self, f"{prefix}_cam")
        gripper = getattr(self, f"{prefix}_gripper")
        if not cam.connect():
            return False
        if not gripper.connect():
            gripper.disconnect()
            return False
        return True

    def _disconnect_side(self, prefix: str) -> None:
        getattr(self, f"{prefix}_cam").disconnect()
        getattr(self, f"{prefix}_gripper").disconnect()

    def _read_side_once(self, prefix: str, copy_frame: bool = True) -> Dict[str, Any]:
        cam = getattr(self, f"{prefix}_cam")
        tracker_device_name = getattr(self, f"{prefix}_tracker_device_name")
        gripper = getattr(self, f"{prefix}_gripper")
        frame, camera_time = cam.read_once(copy_frame=copy_frame, return_time=True)
        pose = self.tracker.read_once(device_name=tracker_device_name) if tracker_device_name else None
        gripper_pos = gripper.read_once()
        observation_time = float(camera_time) if camera_time > 0 else time.time()
        return {
            "time": observation_time,
            "pose": pose,
            "frame": frame,
            "gripper_pos": gripper_pos,
        }

    def _default_sample_hz(self) -> float:
        return float(min(self.left_cam.camera_fps, self.right_cam.camera_fps))

    def _pose_to_array(self, pose: Any) -> np.ndarray:
        if pose is None:
            return np.full((7,), np.nan, dtype=np.float64)

        return np.asarray([*pose.position, *pose.rotation], dtype=np.float64)

    def __del__(self) -> None:
        try:
            self.disconnect()
        except Exception:
            pass
