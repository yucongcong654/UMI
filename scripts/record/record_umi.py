"""
循环录制单臂或双臂 UMI 数据

单臂:
python scripts/record/record_umi.py \
    --camera-device-id 1 \
    --config-path config.json \
    --sample-hz 30

双臂:
python3 scripts/record/record_umi.py \
  --mode bimanual \
  --config-path config.json \
  --sample-hz 30 \
  --tracker-device-name WM0 \
  --right-tracker-device-name WM1 \
  --camera-device-id 0 \
  --right-camera-device-id 1 \
  --gripper-serial-port /dev/ttyACM0 \
  --right-gripper-serial-port /dev/ttyACM1
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("umi.record_umi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="循环录制单臂或双臂 UMI 数据，按回车开始/结束，按 q 退出")
    parser.add_argument("--mode", type=str, choices=("single", "bimanual"), default="single")
    parser.add_argument("--config-path", type=str, default="config.json")
    parser.add_argument("--sample-hz", type=float, default=None)
    parser.add_argument("--tracker-device-name", type=str, default="WM0")
    parser.add_argument("--camera-device-id", type=int, default=0)
    parser.add_argument("--gripper-serial-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--left-tracker-device-name", type=str, default=None)
    parser.add_argument("--right-tracker-device-name", type=str, default="WM1")
    parser.add_argument("--left-camera-device-id", type=int, default=None)
    parser.add_argument("--right-camera-device-id", type=int, default=1)
    parser.add_argument("--left-gripper-serial-port", type=str, default=None)
    parser.add_argument("--right-gripper-serial-port", type=str, default="/dev/ttyACM1")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--wait-for-ready-seconds", type=float, default=5.0)
    return parser.parse_args()


def resolve_output_dir(output_dir: str) -> Path:
    output_path = Path(output_dir)
    if output_path.is_absolute():
        return output_path
    return PROJECT_ROOT / output_path


def next_output_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    date_prefix = datetime.now().strftime("%Y_%m_%d")
    pattern = re.compile(rf"^{re.escape(date_prefix)}_(\d+)\.(?:h5|hdf5)$", re.IGNORECASE)
    max_index = 0
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        max_index = max(max_index, int(match.group(1)))
    return output_dir / f"{date_prefix}_{max_index + 1}.hdf5"


def build_umi(args: argparse.Namespace) -> tuple[Any, str]:
    from src.umi import BimanualUMI, UMI

    if args.mode == "bimanual":
        return (
            BimanualUMI(
                tracker_config=args.config_path,
                left_tracker_device_name=args.left_tracker_device_name or args.tracker_device_name,
                right_tracker_device_name=args.right_tracker_device_name,
                left_camera_device_id=args.left_camera_device_id if args.left_camera_device_id is not None else args.camera_device_id,
                right_camera_device_id=args.right_camera_device_id,
                left_gripper_serial_port=args.left_gripper_serial_port or args.gripper_serial_port,
                right_gripper_serial_port=args.right_gripper_serial_port,
            ),
            "双UMI",
        )
    return (
        UMI(
            tracker_config=args.config_path,
            tracker_device_name=args.tracker_device_name,
            camera_device_id=args.camera_device_id,
            gripper_serial_port=args.gripper_serial_port,
        ),
        "单UMI",
    )


def run() -> int:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    umi, umi_label = build_umi(args)

    logger.info("连接%s...", umi_label)
    if not umi.connect():
        logger.error("连接%s失败", umi_label)
        return 1

    logger.info("%s已连接，录制文件将保存到: %s", umi_label, output_dir)

    try:
        while True:
            user_input = input("按回车开始录制，输入 q 再回车退出: ").strip().lower()
            if user_input == "q":
                break
            if user_input:
                logger.warning("仅支持直接按回车开始录制，或输入 q 再回车退出")
                continue

            output_path = next_output_path(output_dir)
            stop_event = threading.Event()
            exit_after_episode = threading.Event()

            def wait_for_stop() -> None:
                command = input("录制中，按回车结束；输入 q 再回车结束并退出: ").strip().lower()
                if command == "q":
                    exit_after_episode.set()
                stop_event.set()

            waiter = threading.Thread(target=wait_for_stop, daemon=True)
            waiter.start()

            try:
                result = umi.record_episode(
                    file_path=output_path,
                    sample_hz=args.sample_hz,
                    overwrite=False,
                    wait_for_ready_seconds=args.wait_for_ready_seconds,
                    stop_event=stop_event,
                )
            except KeyboardInterrupt:
                stop_event.set()
                raise
            except Exception:
                stop_event.set()
                logger.exception("录制失败: %s", output_path)
                continue

            logger.info("录制完成: %s", result["hdf5_path"])
            logger.info("样本数: %d, 录制时长: %.3fs", result["sample_count"], result["duration"])

            if exit_after_episode.is_set():
                break
    except KeyboardInterrupt:
        logger.info("收到中断信号，结束录制循环")
    finally:
        umi.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
