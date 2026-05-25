# UMI HDF5 数据训练指南（单设备）

本文说明如何使用 `umi.py` 采集的 `data/*.hdf5` 直接训练 diffusion policy（单设备：单相机 + 单 tracker + 单夹爪）。

## 1. 数据格式约定

当前训练数据集类读取每个 HDF5 文件中的这 3 个数据集：

- `frame`: 形状 `[T, H, W, C]`，`uint8`
- `pose`: 形状 `[T, 7]`，`[x, y, z, qx, qy, qz, qw]`
- `gripper_pos`: 形状 `[T]`，夹爪开合量

每个 `.hdf5` 文件会被视为 1 个 episode。

## 2. 训练命令

```bash
nohup python third_party_lib/universal_manipulation_interface-main/train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task=umi_hdf5 \
  task.dataset_dir=/root/autodl-tmp/data \
  hydra.run.dir=/root/autodl-tmp/UMI/outputs/umi_hdf5_run1 > train_log.log &
```

上述命令会将最新模型保存到：

- `/root/autodl-tmp/UMI/outputs/umi_hdf5_run1/checkpoints/latest.ckpt`

- 指定训练输出目录（从而控制 checkpoint 保存位置）：

```bash
hydra.run.dir=/Users/caixinyu/dev/UMI/outputs/umi_hdf5_run1
```

## 5. 关键文件

- 数据集实现：`third_party_lib/universal_manipulation_interface-main/diffusion_policy/dataset/umi_hdf5_dataset.py`
- 任务配置：`third_party_lib/universal_manipulation_interface-main/diffusion_policy/config/task/umi_hdf5.yaml`

## 6. 常见问题

- 报错 “No files matched ...”：
  - 检查 `task.dataset_dir` 和 `task.file_glob`
- 报错 “No valid episodes found ...”：
  - 检查 `pose` 是否含大量 NaN/无效四元数，或 episode 长度太短
- 显存不足：
  - 降低 batch size，例如 `dataloader.batch_size=16 val_dataloader.batch_size=16`
