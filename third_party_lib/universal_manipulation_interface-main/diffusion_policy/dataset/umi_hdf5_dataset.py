import copy
import pathlib
from typing import Dict, Optional

import cv2
import h5py
import numpy as np
import torch
import zarr
from scipy.spatial.transform import Rotation as R
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from umi.common.pose_util import mat_to_pose10d, pose_to_mat


class UmiHdf5Dataset(BaseDataset):
    """
    面向单设备 UMI 的 HDF5 数据集。

    数据流:
    1) 读取 data/*.hdf5（每个文件视为 1 个 episode）
    2) 将原始 xyz + quat 转成 xyz + rotvec
    3) 构造 ReplayBuffer 标准键:
       - camera0_rgb
       - robot0_eef_pos
       - robot0_eef_rot_axis_angle
       - robot0_gripper_width
       - action (7D: xyz + rotvec + gripper)
    4) 通过 SequenceSampler 按 horizon/latency/downsample 采样
    5) __getitem__ 中再将 action/obs 变成模型训练期望表示:
       - pose 由 rotvec 转为 10D (xyz + rot6d)
       - 最终 action 为 10D: xyz + rot6d + gripper
    """
    def __init__(
        self,
        shape_meta: dict,
        dataset_dir: str,
        file_glob: str = "*.hdf5",
        pose_repr: dict = {},
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        repeat_frame_prob: float = 0.0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_duration: Optional[float] = None,
        rotation_correction_y_deg: float = -90.0,
        apply_pose_alignment: bool = True,
        resize_image: bool = True,
        min_episode_length: int = 2,
    ):
        # pose_repr 控制在 __getitem__ 时将位姿转成何种参考系表示
        # 常见值: abs / rel / relative / delta（具体见 convert_pose_mat_rep）
        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get("obs_pose_repr", "relative")
        self.action_pose_repr = self.pose_repr.get("action_pose_repr", "relative")

        # 第一步: 把 HDF5 目录构造成统一 ReplayBuffer（内存 zarr）
        # rotation_correction_y_deg / apply_pose_alignment 保留在接口上，
        # 仅用于兼容已有配置；实时对齐/校准已经迁移到推理脚本中处理。
        replay_buffer = self._build_replay_buffer_from_hdf5(
            shape_meta=shape_meta,
            dataset_dir=dataset_dir,
            file_glob=file_glob,
            resize_image=resize_image,
            min_episode_length=min_episode_length,
        )

        # 当前实现是单设备，因此机器人数量固定为 1
        self.num_robot = 1
        rgb_keys = []
        lowdim_keys = []
        key_horizon = {}
        key_down_sample_steps = {}
        key_latency_steps = {}

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            # 根据类型拆分 rgb / low_dim 键
            this_type = attr.get("type", "low_dim")
            if this_type == "rgb":
                rgb_keys.append(key)
            elif this_type == "low_dim":
                lowdim_keys.append(key)

            # 每个观测键都从 shape_meta 中读取采样参数
            key_horizon[key] = shape_meta["obs"][key]["horizon"]
            key_latency_steps[key] = shape_meta["obs"][key]["latency_steps"]
            key_down_sample_steps[key] = shape_meta["obs"][key]["down_sample_steps"]

        # action 的 horizon/latency/downsample 单独读取
        key_horizon["action"] = shape_meta["action"]["horizon"]
        key_latency_steps["action"] = shape_meta["action"]["latency_steps"]
        key_down_sample_steps["action"] = shape_meta["action"]["down_sample_steps"]

        # 构建训练/验证 episode mask
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask

        # 统一使用官方 SequenceSampler，保证和 diffusion_policy 训练链路一致
        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration,
        )

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False

    @staticmethod
    def _resize_frame(frame: np.ndarray, target_hwc: tuple[int, int, int]) -> np.ndarray:
        """
        将单帧图像 resize 到 shape_meta 约定分辨率。
        target_hwc 来自 camera0_rgb.shape 的 C/H/W 转换。
        """
        target_h, target_w, target_c = target_hwc
        if frame.ndim != 3:
            raise ValueError(f"Expected frame ndim=3, got {frame.ndim}")
        if frame.shape[2] != target_c:
            raise ValueError(f"Expected frame channels={target_c}, got {frame.shape[2]}")
        if frame.shape[0] == target_h and frame.shape[1] == target_w:
            return frame
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    @classmethod
    def _build_replay_buffer_from_hdf5(
        cls,
        shape_meta: dict,
        dataset_dir: str,
        file_glob: str,
        resize_image: bool,
        min_episode_length: int,
    ) -> ReplayBuffer:
        """
        将目录中的 HDF5 文件转换为 ReplayBuffer。

        每个 HDF5 文件被视为一个 episode，至少包含:
        - pose: [T, 7] -> xyz + quat(xyzw)
        - gripper_pos: [T]
        - frame: [T, H, W, C]
        """
        # 支持传入相对路径/~，统一展开为绝对路径
        dataset_path = pathlib.Path(dataset_dir).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset_dir does not exist: {dataset_path}")

        # 用 file_glob 控制文件过滤，例如 *.hdf5 / *.h5
        file_paths = sorted(dataset_path.rglob(file_glob))
        if not file_paths:
            raise FileNotFoundError(f"No files matched {file_glob} under {dataset_path}")

        # shape_meta 中 rgb 是 [C,H,W]，这里转成 OpenCV 习惯的 [H,W,C]
        camera_shape = shape_meta["obs"]["camera0_rgb"]["shape"]
        target_hwc = (int(camera_shape[1]), int(camera_shape[2]), int(camera_shape[0]))

        # 直接使用内存后端，便于训练随机采样；大数据集可再扩展到磁盘缓存
        replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())

        for file_path in tqdm(file_paths, desc="Loading HDF5 episodes"):
            with h5py.File(file_path, "r") as h5f:
                # 强约束输入字段，避免 silent bug
                required_keys = ("pose", "gripper_pos", "frame")
                for key in required_keys:
                    if key not in h5f:
                        raise KeyError(f"{file_path} missing dataset: {key}")

                poses = np.asarray(h5f["pose"], dtype=np.float64)
                gripper_pos = np.asarray(h5f["gripper_pos"], dtype=np.float64)
                frames = np.asarray(h5f["frame"], dtype=np.uint8)

                # 三路数据长度不一致时取最短，避免越界
                n_steps = min(len(poses), len(gripper_pos), len(frames))
                if n_steps < min_episode_length:
                    continue

                out_pos = []
                out_rotvec = []
                out_gripper = []
                out_frames = []

                for i in range(n_steps):
                    # pose 应为 7D: xyz + quat
                    pose_i = poses[i]
                    if pose_i.shape[-1] != 7:
                        continue
                    if not np.all(np.isfinite(pose_i)):
                        continue

                    position = pose_i[:3]
                    quaternion_xyzw = pose_i[3:7]
                    quat_norm = np.linalg.norm(quaternion_xyzw)
                    if quat_norm <= np.finfo(np.float64).eps:
                        continue
                    # 始终归一化四元数，避免累计数值误差
                    quaternion_xyzw = quaternion_xyzw / quat_norm

                    # 直接保留数据集中的原始位姿；实时对齐/校准在推理阶段完成。
                    rotvec = R.from_quat(quaternion_xyzw).as_rotvec()

                    frame = frames[i]
                    if resize_image:
                        frame = cls._resize_frame(frame=frame, target_hwc=target_hwc)
                    if not np.all(np.isfinite(frame)):
                        continue

                    gripper_i = gripper_pos[i]
                    if not np.isfinite(gripper_i):
                        # gripper 缺失时做简单前向填充；首帧缺失则置 0
                        if out_gripper:
                            gripper_i = out_gripper[-1]
                        else:
                            gripper_i = 0.0

                    out_pos.append(position.astype(np.float32))
                    out_rotvec.append(rotvec.astype(np.float32))
                    out_gripper.append(float(gripper_i))
                    out_frames.append(frame.astype(np.uint8))

                if len(out_pos) < min_episode_length:
                    continue

                # 将本 episode 拼成训练标准数组
                pos_arr = np.asarray(out_pos, dtype=np.float32)
                rotvec_arr = np.asarray(out_rotvec, dtype=np.float32)
                gripper_arr = np.asarray(out_gripper, dtype=np.float32).reshape(-1, 1)
                frame_arr = np.asarray(out_frames, dtype=np.uint8)
                # action 基础形态: 7D = xyz + rotvec + gripper
                action_arr = np.concatenate([pos_arr, rotvec_arr, gripper_arr], axis=-1).astype(np.float32)

                # 与 UmiDataset/SequenceSampler 兼容的键名
                replay_buffer.add_episode(
                    {
                        "camera0_rgb": frame_arr,
                        "robot0_eef_pos": pos_arr,
                        "robot0_eef_rot_axis_angle": rotvec_arr,
                        "robot0_gripper_width": gripper_arr,
                        "action": action_arr,
                    }
                )

        if replay_buffer.n_episodes == 0:
            raise RuntimeError(f"No valid episodes found in {dataset_path}")
        return replay_buffer

    def get_validation_dataset(self):
        """
        生成验证集视图:
        - 共享同一个 replay_buffer（不重复占用内存）
        - 仅替换 sampler 的 episode_mask
        """
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """
        统计并构建归一化器:
        - action: pos 用 range，rot6d 用 identity，gripper 用 range
        - obs:    pos 用 range，rot6d 用 identity，gripper 用 range
        - rgb:    identity（后续在网络输入端按图像规则处理）
        """
        normalizer = LinearNormalizer()

        # 通过 DataLoader 走完整采样逻辑，确保统计口径与训练一致
        data_cache = {key: [] for key in self.lowdim_keys + ["action"]}
        # 统计归一化时无需读取图像，可显著降低开销
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=8,
        )
        for batch in tqdm(dataloader, desc="Iterating dataset to get normalization"):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch["obs"][key]))
            data_cache["action"].append(copy.deepcopy(batch["action"]))
        self.sampler.ignore_rgb(False)

        # 拼接并检查维度: [B, T, D]
        for key in data_cache:
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            b_size, t_size, d_size = data_cache[key].shape
            # 是否按时间维独立归一化由配置控制
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(b_size * t_size, d_size)

        # action = [xyz, rot6d, gripper] -> [3, 6, 1]
        action_normalizers = [
            get_range_normalizer_from_stat(array_to_stats(data_cache["action"][..., 0:3])),
            get_identity_normalizer_from_stat(array_to_stats(data_cache["action"][..., 3:9])),
            get_range_normalizer_from_stat(array_to_stats(data_cache["action"][..., 9:10])),
        ]
        normalizer["action"] = concatenate_normalizer(action_normalizers)

        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])
            if key.endswith("pos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("rot_axis_angle"):
                # 注意: 这里实际已经是 rot6d（名字沿用了原 UMI 命名）
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("gripper_width"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f"Unsupported lowdim key: {key}")
            normalizer[key] = this_normalizer

        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        采样并打包单条训练样本。

        返回:
        {
          "obs": {
             camera0_rgb: [T, C, H, W] float32 in [0,1]
             robot0_eef_pos: [T, 3]
             robot0_eef_rot_axis_angle: [T, 6]  # 实际是 rot6d
             robot0_gripper_width: [T, 1]
          },
          "action": [Ta, 10]  # xyz + rot6d + gripper
        }
        """
        # 限制底层线程池，避免 dataloader 多进程下线程过量
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        data = self.sampler.sample_sequence(idx)

        obs_dict = {}
        for key in self.rgb_keys:
            if key not in data:
                continue
            # 图像从 THWC -> TCHW，并缩放到 [0,1]
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.0
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]

        # 将 obs 当前姿态和 future action 都转成 pose matrix
        pose_mat = pose_to_mat(
            np.concatenate(
                [
                    obs_dict["robot0_eef_pos"],
                    obs_dict["robot0_eef_rot_axis_angle"],
                ],
                axis=-1,
            )
        )
        action_mat = pose_to_mat(data["action"][..., :6])

        # 根据配置将姿态表达转成 abs/relative/delta 等
        obs_pose_mat = convert_pose_mat_rep(
            pose_mat,
            base_pose_mat=pose_mat[-1],
            pose_rep=self.obs_pose_repr,
            backward=False,
        )
        action_pose_mat = convert_pose_mat_rep(
            action_mat,
            base_pose_mat=pose_mat[-1],
            pose_rep=self.action_pose_repr,
            backward=False,
        )

        # mat -> 10D (xyz + rot6d)
        obs_pose = mat_to_pose10d(obs_pose_mat)
        action_pose = mat_to_pose10d(action_pose_mat)
        # gripper 单独拼接，最终 action 维度为 10
        action_gripper = data["action"][..., 6:7]

        obs_dict["robot0_eef_pos"] = obs_pose[:, :3]
        obs_dict["robot0_eef_rot_axis_angle"] = obs_pose[:, 3:]
        data["action"] = np.concatenate([action_pose, action_gripper], axis=-1)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
        return torch_data
