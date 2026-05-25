#!/usr/bin/env python3
"""
Minimal UMI-style real evaluation loop for a single OpenArm follower.

This script keeps the core structure of `eval_real_umi.py`:
1. Load a UMI checkpoint.
2. Read real observations from one camera + one robot.
3. Convert observations to the UMI policy input format.
4. Run policy inference.
5. Convert predicted end-effector actions into joint targets with IK.
6. Send joint commands through LeRobot's OpenArm follower interface.

lerobot-setup-can --mode=setup --interfaces=can0,can1

python scripts/eval/eval_openarm_umi.py \
    --input outputs/umi_hdf5_run2/checkpoints/latest.ckpt \
    --urdf-path src/kinematics/description/openarm/urdf/openarm_bimanual.urdf \
    --camera-device 1 \
    --side left \
    --robot-id my_openarm_umi_left2 \
    --port can1
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
UMI_ROOT = PROJECT_ROOT / "third_party_lib" / "universal_manipulation_interface-main"
LEROBOT_SRC = PROJECT_ROOT / "third_party_lib" / "lerobot-0.5.1" / "src"
for path in (PROJECT_ROOT, UMI_ROOT, LEROBOT_SRC):
    sys.path.append(str(path))
os.chdir(PROJECT_ROOT)

import click
import dill
import hydra
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from openarm_umi_env import OpenArmUmiEnv, make_camera, make_robot
from openarm_umi_episode import run_episode
from src.kinematics.openarm_solver import OpenArmIK

OmegaConf.register_new_resolver("eval", eval, replace=True)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_policy(ckpt_path: str, device: torch.device) -> tuple[Any, Any]:
    ckpt = ckpt_path
    if not ckpt.endswith(".ckpt"):
        ckpt = os.path.join(ckpt, "checkpoints", "latest.ckpt")

    with open(ckpt, "rb") as f:
        payload = torch.load(f, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = 16
    policy.eval().to(device)
    return policy, cfg


@click.command()
@click.option("--input", "-i", "input_path", required=True, help="UMI checkpoint path.")
@click.option("--urdf-path", required=True, help="URDF used by the OpenArm IK solver.")
@click.option("--port", required=True, help="OpenArm CAN port, e.g. can0.")
@click.option("--side", type=click.Choice(["left", "right"]), default="left", show_default=True)
@click.option("--robot-id", required=True, show_default=True)
@click.option("--camera-device", default="0", show_default=True, help="OpenCV index/path.")
@click.option(
    "--camera-width",
    default=None,
    type=int,
    help="OpenCV capture width. Defaults to the camera's native width.",
)
@click.option(
    "--camera-height",
    default=None,
    type=int,
    help="OpenCV capture height. Defaults to the camera's native height.",
)
@click.option("--camera-fps", default=60, type=int, show_default=True)
@click.option("--steps-per-inference", "-si", default=6, type=int, show_default=True)
@click.option("--frequency", "-f", default=10.0, type=float, show_default=True)
@click.option("--max-duration", "-md", default=13.0, type=float, show_default=True)
@click.option("--device", default="auto", show_default=True, help="auto/cpu/cuda")
@click.option("--can-interface", default="socketcan", show_default=True)
@click.option("--max-relative-target", default=6.0, type=float, show_default=True)
@click.option(
    "--gripper-max-relative-target",
    default=65,
    type=float,
    help="Override `max_relative_target` for the gripper only, while keeping the arm joints unchanged.",
)
@click.option(
    "--gripper-open-deg",
    default=-65.0,
    type=float,
    show_default=True,
    help="Follower gripper open position in degrees. `robot_obs['gripper.pos']` is reported in degrees.",
)
@click.option("--gripper-closed-deg", default=0.0, type=float, show_default=True)
@click.option("--policy-gripper-open-value", default=1.0, type=float, show_default=True)
@click.option("--policy-gripper-closed-value", default=-0.72449, type=float, show_default=True)
@click.option(
    "--action-interp-alpha",
    default=1.0,
    type=float,
    show_default=True,
    help="Interpolate motor targets with the previous command; 1.0 disables smoothing.",
)
@click.option(
    "--visualize-camera/--no-visualize-camera",
    default=True,
    show_default=True,
    help="Show a live OpenCV preview window while the evaluation loop is running.",
)
@click.option(
    "--tracker-rotation-correction-y-deg",
    default=90,
    type=float,
    help="Override checkpoint tracker rotation correction around Y for quick calibration.",
)
@click.option("--no-calibrate", is_flag=False, default=True, help="Skip automatic LeRobot calibration.")
def main(
    input_path: str,
    urdf_path: str,
    port: str,
    side: str,
    robot_id: str,
    camera_device: str,
    camera_width: int | None,
    camera_height: int | None,
    camera_fps: int,
    steps_per_inference: int,
    frequency: float,
    max_duration: float,
    device: str,
    can_interface: str,
    max_relative_target: float,
    gripper_max_relative_target: float | None,
    gripper_open_deg: float,
    gripper_closed_deg: float,
    policy_gripper_open_value: float,
    policy_gripper_closed_value: float,
    action_interp_alpha: float,
    visualize_camera: bool,
    tracker_rotation_correction_y_deg: float | None,
    no_calibrate: bool,
) -> None:
    torch_device = resolve_device(device)
    policy, cfg = load_policy(input_path, torch_device)
    action_pose_repr = cfg.task.pose_repr.action_pose_repr
    dataset_cfg = getattr(cfg.task, "dataset", {})
    rotation_correction_y_deg = float(
        tracker_rotation_correction_y_deg
        if tracker_rotation_correction_y_deg is not None
        else getattr(dataset_cfg, "rotation_correction_y_deg", 0.0)
    )
    print(f"Using tracker rotation correction around Y only: {rotation_correction_y_deg} deg")
    if gripper_max_relative_target is None:
        robot_max_relative_target: float | dict[str, float] = max_relative_target
    else:
        robot_max_relative_target = {f"joint_{i}": max_relative_target for i in range(1, 8)}
        robot_max_relative_target["gripper"] = float(gripper_max_relative_target)
        print(
            "Using per-motor max_relative_target "
            f"(joints={max_relative_target}, gripper={gripper_max_relative_target})"
        )

    ik_solver = OpenArmIK(
        urdf_path=urdf_path,
        left_ee="openarm_left_hand",
        right_ee="openarm_right_hand",
    )
    robot = make_robot(
        robot_id=robot_id,
        side=side,
        port=port,
        can_interface=can_interface,
        max_relative_target=robot_max_relative_target,
    )
    camera = make_camera(
        camera_device=camera_device,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fps=camera_fps,
    )
    robot.connect(calibrate=not no_calibrate)
    if not camera.connect():
        robot.disconnect()
        raise RuntimeError(f"Failed to connect OpenCVCam device: {camera_device}")

    try:
        env = OpenArmUmiEnv(
            robot=robot,
            camera=camera,
            ik_solver=ik_solver,
            side=side,
            shape_meta=cfg.task.shape_meta,
            gripper_open_deg=gripper_open_deg,
            gripper_closed_deg=gripper_closed_deg,
            policy_gripper_open_value=policy_gripper_open_value,
            policy_gripper_closed_value=policy_gripper_closed_value,
            action_interp_alpha=action_interp_alpha,
            visualize_camera=visualize_camera,
            tracker_rotation_correction_y_deg=rotation_correction_y_deg,
        )
        print("Moving robot to the default OpenArm pose...")
        env.move_to_default_pose()
        print("Press ENTER to start an episode. Input 'q' then ENTER to exit.")
        episode_id = 0

        while True:
            user_input = input("> ").strip().lower()
            if user_input == "q":
                break

            steps, duration_s = run_episode(
                policy=policy,
                cfg=cfg,
                env=env,
                action_pose_repr=action_pose_repr,
                device=torch_device,
                steps_per_inference=steps_per_inference,
                frequency=frequency,
                max_duration=max_duration,
            )
            print(
                f"Episode {episode_id} finished: "
                f"{steps} steps, {duration_s:.2f}s."
            )
            episode_id += 1
            print("Press ENTER to start the next episode. Input 'q' then ENTER to exit.")
    finally:
        try:
            if 'env' in locals():
                print("Moving robot to the zero joint pose before disconnecting...")
                env.move_to_zero_pose()
        finally:
            if 'env' in locals():
                env.close()
            camera.disconnect()
            robot.disconnect()


if __name__ == "__main__":
    main()
