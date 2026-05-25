"""
从USB鱼眼相机里读取相机图像
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("umi.opencv_cam")


class OpenCVCam:
    def __init__(
        self,
        camera_width: int | None = 640,
        camera_height: int | None = 480,
        camera_fps: int | None = 30,
        device_id: int | str = 0,
        read_thread_fps: int = 120,
        auto_reconnect: bool = True,
    ) -> None:
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.device_id = device_id
        self.read_thread_fps = read_thread_fps
        self.auto_reconnect = auto_reconnect

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self._stop_thread = True
        self._read_thread: Optional[threading.Thread] = None
        self._frame_lock = threading.Lock()
        self._has_frame = False
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time = 0.0

    def _configure_capture(self, cap: cv2.VideoCapture) -> None:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if self.camera_width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        if self.camera_height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        if self.camera_fps is not None:
            cap.set(cv2.CAP_PROP_FPS, self.camera_fps)

    def connect(self) -> bool:
        if self.is_connected:
            return True
        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(0)
        cap = cv2.VideoCapture(self.device_id)
        self._configure_capture(cap)
        if not cap.isOpened():
            cap.release()
            logger.error("无法打开相机，设备ID: %s", self.device_id)
            return False

        self.cap = cap
        self.is_connected = True
        self._stop_thread = False
        self._start_read_thread()
        logger.info("相机连接成功，设备ID: %s", self.device_id)
        return True

    def disconnect(self) -> None:
        self._stop_thread = True
        self._stop_read_thread()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        logger.info("相机已断开，设备ID: %s", self.device_id)

    def _start_read_thread(self) -> None:
        if self._read_thread is not None and self._read_thread.is_alive():
            return
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()

    def _stop_read_thread(self) -> None:
        if self._read_thread is not None and self._read_thread.is_alive():
            self._read_thread.join(timeout=1.0)
        self._read_thread = None

    def _read_loop(self) -> None:
        interval = 1.0 / max(self.read_thread_fps, 1)
        while not self._stop_thread:
            if self.cap is None:
                if not self.auto_reconnect or not self._reconnect_once():
                    time.sleep(interval)
                continue

            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self._frame_lock:
                    self._has_frame = True
                    self._last_frame = frame
                    self._last_frame_time = time.time()
            else:
                self._has_frame = False
                if self.auto_reconnect:
                    self._reconnect_once()
                else:
                    time.sleep(interval)
            time.sleep(interval)

    def _reconnect_once(self) -> bool:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cap = cv2.VideoCapture(self.device_id)
        self._configure_capture(cap)
        if not cap.isOpened():
            cap.release()
            self.is_connected = False
            return False
        self.cap = cap
        self.is_connected = True
        return True

    def read_once(
        self,
        copy_frame: bool = True,
        return_time: bool = False,
    ) -> Optional[np.ndarray] | tuple[Optional[np.ndarray], float]:
        with self._frame_lock:
            if not self._has_frame or self._last_frame is None:
                return (None, 0.0) if return_time else None
            frame = self._last_frame.copy() if copy_frame else self._last_frame
            frame_time = float(self._last_frame_time)
            if return_time:
                return frame, frame_time
            return frame

    def get_camera_info(self) -> Dict[str, Any]:
        if self.cap is not None and self.is_connected:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        else:
            width = self.camera_width
            height = self.camera_height
            fps = float(self.camera_fps)
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "device_id": self.device_id,
            "read_thread_fps": self.read_thread_fps,
            "auto_reconnect": self.auto_reconnect,
            "is_connected": self.is_connected,
            "last_frame_time": self._last_frame_time,
        }

    def get_device_info(self, device_name: Optional[str] = None) -> Dict[str, Any]:
        return self.get_camera_info()

    def get_devices(self) -> List[str]:
        if not self.is_connected:
            return []
        return [str(self.device_id)]

    def close(self) -> None:
        self.disconnect()

    def __del__(self) -> None:
        self.disconnect()

def find_available_cameras(max_index: int = 20) -> List[int]:
    available: List[int] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available.append(index)
        cap.release()
    return available
