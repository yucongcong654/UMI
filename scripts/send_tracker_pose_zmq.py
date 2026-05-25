"""
通过 ZMQ 发送实时 tracker 位姿。

发送格式参考 calibrate_ur.py：
[x, y, z, rx, ry, rz]

示例:
python scripts/send_tracker_pose_zmq.py --config-path config.json --device-name WM0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.pose_utils import align_tracker_pose, apply_rotation_correction_y

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("umi.send_tracker_pose_zmq")

DEFAULT_TOPIC = "tracker_pose"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="通过 ZMQ 发送实时 tracker 位姿")
    parser.add_argument("--config-path", type=str, default="config.json")
    parser.add_argument("--device-name", type=str, default="WM0")
    parser.add_argument("--endpoint", type=str, default="tcp://192.168.31.90:5555")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC)
    parser.add_argument("--socket-mode", type=str, choices=("bind", "connect"), default="bind")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--startup-wait", type=float, default=0.2, help="socket 建立后等待订阅者连接的秒数")
    parser.add_argument("--rotation-correction-y-deg", type=float, default=-90.0, help="Y 轴额外旋转校准角度 (度)")
    return parser.parse_args()


def build_payload(
    device_name: str,
    timestamp: float,
    position: np.ndarray,
    quaternion: np.ndarray,
    pose_posrotvec: np.ndarray,
) -> dict[str, object]:
    return {
        "device_name": device_name,
        "timestamp": float(timestamp),
        "pose": pose_posrotvec.tolist(),
        "pose_quat": [float(x) for x in quaternion],
        "position": [float(x) for x in position],
        "pose_format": "xyz_rotvec",
        "source_pose_format": "xyz_quat",
    }


def create_socket(context: object, endpoint: str, socket_mode: str) -> object:
    socket = context.socket(1)
    if socket_mode == "bind":
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)
    return socket


def main() -> None:
    args = parse_args()
    if args.hz <= 0:
        raise ValueError("hz 必须大于 0")

    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError("未安装 pyzmq，请先执行 `pip install -r requirements.txt`") from exc

    try:
        from src.devices.tracker import ViveTracker
    except ImportError as exc:
        raise RuntimeError("未安装 tracker 依赖，请先确认 pysurvive 环境可用") from exc

    context = zmq.Context()
    socket = create_socket(context=context, endpoint=args.endpoint, socket_mode=args.socket_mode)
    tracker = ViveTracker(config_path=str(Path(args.config_path).expanduser().resolve()))

    try:
        if args.startup_wait > 0:
            time.sleep(args.startup_wait)

        if not tracker.connect():
            raise RuntimeError("连接 tracker 失败")

        logger.info(
            "开始发送位姿: device=%s endpoint=%s hz=%.2f rotation_correction_y=%.2f",
            args.device_name,
            args.endpoint,
            args.hz,
            args.rotation_correction_y_deg,
        )

        last_timestamp = None
        sleep_time = 1.0 / args.hz

        while True:
            pose = tracker.read_once(device_name=args.device_name)
            if pose is None:
                time.sleep(sleep_time)
                continue

            if last_timestamp == pose.timestamp:
                time.sleep(sleep_time)
                continue

            aligned_pose = align_tracker_pose(position=pose.position, quaternion=pose.rotation)
            if aligned_pose is None:
                time.sleep(sleep_time)
                continue
            aligned_position, aligned_quaternion, _ = aligned_pose

            # 应用额外旋转校准 (保持位置不变，仅修改旋转：R' = C * R * C^-1)
            aligned_quaternion = apply_rotation_correction_y(
                quaternion=aligned_quaternion,
                correction_y_deg=args.rotation_correction_y_deg,
            )
            pose_posrotvec = R.from_quat(aligned_quaternion).as_rotvec()

            payload = build_payload(
                device_name=args.device_name,
                timestamp=float(pose.timestamp),
                position=aligned_position,
                quaternion=aligned_quaternion,
                pose_posrotvec=pose_posrotvec,
            )
            socket.send_multipart(
                [
                    args.topic.encode("utf-8"),
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                ]
            )
            last_timestamp = pose.timestamp
            time.sleep(sleep_time)
    finally:
        tracker.disconnect()
        socket.close(0)
        context.term()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("用户中断发送")
