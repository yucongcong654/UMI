接口人: @高圆寺

# UMI

## 环境划分

- 采集环境：`Python 3.9`，用于相机、tracker、夹爪等数据采集。
- 训练/推理环境：`Python 3.10`，用于模型训练与推理，建议与采集环境分开创建。

## 采集环境（Python 3.9）

### 1. 创建环境

```shell
conda create -n umi-collect python=3.9 -y
conda activate umi-collect
```

### 2. 安装基础依赖

```shell
sudo apt update
sudo apt install build-essential zlib1g-dev libx11-dev libusb-1.0-0-dev freeglut3-dev liblapacke-dev libopenblas-dev libatlas-base-dev cmake
sudo apt install libgtk-3-dev
```

### 3. 安装 `libsurvive`

```shell
cd third_party_lib/libsurvive-master
sudo cp ./useful_files/81-vive.rules /etc/udev/rules.d/
# 连接无线接收器
sudo udevadm control --reload-rules && sudo udevadm trigger # 执行完这步后，如电脑上插有无线接收器请将其拔插一遍。
make
```

### 4. 安装 Python 依赖

```shell
pip install wxPython==4.2.1 --no-build-isolation
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 训练/推理环境（Python 3.10）

### 1. 创建独立环境

```shell
conda create -n umi-train python=3.10 -y
conda activate umi-train
```

### 2. 安装训练/推理依赖

- 训练/推理依赖建议参考 `third_party_lib/universal_manipulation_interface-main/conda_environment.yaml` 单独安装。
- 训练命令与数据格式说明见 `scripts/training/TRAIN_UMI_DIFFUSION_POLICY.md`。

# 使用指南
Ref: https://dcnk1u1qkxw1.feishu.cn/docx/RhxTd7U2soCXnPxJjS4cX4tFnjd
