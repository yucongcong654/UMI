from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from umi.common.precise_sleep import precise_wait
from umi.real_world.real_inference_util import get_real_umi_action

from openarm_umi_env import OpenArmUmiEnv, normalize_env_action_sequence


def get_policy_rgb_obs_dict(env_obs: dict[str, np.ndarray], shape_meta: dict[str, Any]) -> dict[str, np.ndarray]:
    obs_dict_np: dict[str, np.ndarray] = {}
    for key, attr in shape_meta["obs"].items():
        if attr.get("type", "low_dim") != "rgb":
            continue
        images = np.asarray(env_obs[key], dtype=np.float32)
        if images.ndim != 4:
            raise ValueError(f"Expected RGB observation `{key}` with shape [T, H, W, C], got {images.shape}")
        obs_dict_np[key] = np.moveaxis(images, -1, 1)
    if not obs_dict_np:
        raise ValueError("The checkpoint does not declare any RGB observation keys.")
    return obs_dict_np


def run_episode(
    policy: Any,
    cfg: Any,
    env: OpenArmUmiEnv,
    action_pose_repr: str,
    device: torch.device,
    steps_per_inference: int,
    frequency: float,
    max_duration: float,
) -> tuple[int, float]:
    dt = 1.0 / frequency
    start_monotonic = time.monotonic()
    eval_t_start = time.time()
    iter_idx = 0
    policy.reset()
    action_exec_latency = 0.01

    while True:
        cycle_end = start_monotonic + (iter_idx + steps_per_inference) * dt
        obs = env.get_obs()
        obs_timestamps = obs["timestamp"]
        with torch.no_grad():
            obs_dict_np = get_policy_rgb_obs_dict(obs, cfg.task.shape_meta)
            obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            result = policy.predict_action(obs_dict)
            raw_action = result["action_pred"][0].detach().cpu().numpy()
            env_action = normalize_env_action_sequence(
                get_real_umi_action(raw_action, obs, action_pose_repr)
            )

        action_timestamps = np.arange(len(env_action), dtype=np.float64) * dt + float(obs_timestamps[-1])
        curr_time = time.time()
        is_new = action_timestamps > (curr_time + action_exec_latency)
        if np.sum(is_new) == 0:
            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
            action_timestamps = np.array([eval_t_start + next_step_idx * dt], dtype=np.float64)
            env_action = env_action[[-1]]
        else:
            env_action = env_action[is_new]
            action_timestamps = action_timestamps[is_new]
        env.exec_actions(env_action, action_timestamps)

        if time.monotonic() - start_monotonic >= max_duration:
            break

        precise_wait(cycle_end)
        iter_idx += steps_per_inference

    return iter_idx, (time.monotonic() - start_monotonic)
