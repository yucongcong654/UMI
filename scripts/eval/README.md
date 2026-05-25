# OpenArm UMI Eval 说明

这个目录提供了一个最小化的单臂 OpenArm + UMI 实机推理闭环。

目录中的脚本职责如下：

- `eval_openarm_umi.py`
  - 评测入口脚本。
  - 负责加载 checkpoint、初始化相机和机械臂、构建 IK 求解器，并循环执行 episode。
- `openarm_umi_episode.py`
  - 单个 episode 的推理与调度逻辑。
  - 负责获取观测、调用策略、校准策略输出、生成执行时间戳，并把动作交给环境执行。
- `openarm_umi_env.py`
  - OpenArm 到 UMI 接口的适配层。
  - 负责把真实机器人观测包装成 UMI 期望的格式，并把末端位姿动作转换成关节角命令发送给机器人。

## 整体执行流程

一次推理循环的大致链路如下：

1. `eval_openarm_umi.py` 加载模型、配置、相机、机械臂和 `OpenArmIK`。
2. 构造 `OpenArmUmiEnv`，并先把机械臂移动到默认姿态。
3. `run_episode()` 周期性读取观测 `env.get_obs()`。
4. 观测经过 `get_real_umi_obs_dict()` 转成模型输入。
5. 策略输出单步或多步动作 `action_pred`。
6. 动作先在 `calibrate_policy_action_sequence()` 中做坐标系相关校准。
7. 再通过 `get_real_umi_action()` 把策略输出的位姿表示恢复成环境执行的末端位姿动作。
8. `OpenArmUmiEnv.exec_action()` 调用 IK，把末端目标位姿解成关节角。
9. 关节角和夹爪角度被发送到 OpenArm follower。

## 模型输出的动作格式

当前代码按单机器人 `10` 维动作解释策略输出：

- 前 `9` 维：`pose10d`
  - `3` 维平移
  - `6` 维旋转 `rot6d`
- 第 `10` 维：夹爪值

在 `openarm_umi_episode.py` 中：

- `raw_action = result["action_pred"][0].detach().cpu().numpy()`
- 然后调用 `calibrate_policy_action_sequence(raw_action, ...)`

`pose10d` 与位姿矩阵的关系来自 UMI 的 `pose_util.py`：

- `pose10d_to_mat()`：`pose10d -> 4x4` 齐次变换矩阵
- `mat_to_pose10d()`：`4x4` 齐次变换矩阵 `-> pose10d`

## 相对位姿变成关节角的处理链路

这是最关键的一段。

### 1. 策略动作校准

`openarm_umi_episode.py` 中的 `calibrate_policy_action_sequence()` 会先把策略输出的 `pose10d` 还原为位姿矩阵，再拆成位置和四元数，接着执行两类可选处理：

- `align_tracker_pose()`
  - 把追踪器坐标系中的位姿对齐到当前机器人控制使用的坐标系。
- `apply_rotation_correction_y()`
  - 对姿态附加一个绕 Y 轴的固定旋转修正。

处理完成后，再把位姿重新编码回 `pose10d`，保留原始夹爪值。

这一层的作用可以理解为：

- 先把模型输出的动作放到“机器人真正使用的坐标系”里。

### 2. 相对位姿恢复成绝对末端目标位姿

校准后的动作会传给 UMI 的 `get_real_umi_action()`。

这个函数会：

1. 从当前观测里拿到机器人当前末端位姿：
   - `robot0_eef_pos[-1]`
   - `robot0_eef_rot_axis_angle[-1]`
2. 把当前末端位姿转成 `base_pose_mat`
3. 把策略输出的 `action_pose10d` 转成 `action_pose_mat`
4. 调用 `convert_pose_mat_rep(..., backward=True)` 做推理阶段的位姿表示反变换

如果配置里的 `action_pose_repr` 是 `relative`，核心公式是：

```python
target_pose_mat = current_eef_pose_mat @ relative_action_pose_mat
```

也就是说：

- 模型输出的是“相对当前末端”的目标位姿
- 真正执行前，要结合当前真实末端姿态，把它恢复成绝对目标位姿

恢复完成后，UMI 会把位姿矩阵转成环境侧使用的 `6D pose`：

- `xyz`
- `rotvec`

再与夹爪值拼接成 `7D` 环境动作：

```python
[x, y, z, rx, ry, rz, gripper]
```

### 3. 环境动作转成 SE(3)

`openarm_umi_env.py` 中的 `exec_action()` 接收到 `7D` 动作后：

1. 取前 `6` 维末端位姿
2. 调用 `pose_to_se3()` 转成 `jaxlie.SE3`

这里旋转使用的是轴角 `rotvec`。

### 4. IK 求解得到关节角

随后 `exec_action()` 会再次读取当前机器人状态，并构造当前全身关节配置 `q_current`。

然后根据左右臂分别调用：

- `ik_solver.solve(target_L=target_se3, target_R=None, q_current=...)`
- 或
- `ik_solver.solve(target_L=None, target_R=target_se3, q_current=...)`

IK 输出为新的关节配置 `q_next`，单位是弧度。

### 5. 关节角转成电机命令

`q_next` 会继续被转换成 follower 接口使用的电机目标：

- `joint_1.pos`
- `joint_2.pos`
- ...
- `joint_7.pos`

每个关节都会执行：

```python
np.rad2deg(q_next[q_idx])
```

也就是把 IK 求出的弧度转成角度。

与此同时：

- 夹爪动作会通过 `_policy_gripper_to_deg()` 从策略值映射到实际夹爪角度

最终一起组成发给 `self.robot.send_action(...)` 的命令字典。

## 一条完整的动作链

可以把整条链路简化成下面这样：

```text
策略输出 action_pred
-> pose10d + gripper
-> 坐标系对齐 / 旋转修正
-> 相对位姿恢复为绝对末端目标位姿
-> [x, y, z, rotvec, gripper]
-> jaxlie.SE3
-> IK.solve(...)
-> q_next
-> rad2deg
-> joint_i.pos / gripper.pos
-> OpenArm follower
```

## 推理频率和执行频率

这套代码里，推理频率和执行频率不是两套独立时钟，而是共享同一个控制周期 `dt`：

```python
dt = 1.0 / frequency
```

其中：

- `frequency`
  - 是环境动作的目标执行频率
  - 例如 `frequency = 60`，表示目标执行周期约为 `1 / 60 s`
- `steps_per_inference`
  - 是每次策略推理后，计划连续执行多少步动作

因此可以直接得到：

```text
执行频率 = frequency
推理频率 = frequency / steps_per_inference
```

### 调度方式

`openarm_umi_episode.py` 每次推理后，会得到一段动作序列 `env_action`，然后按固定步长 `dt` 给这些动作分配执行时间：

```python
action_timestamps = obs_timestamps[-1] + [0, 1, 2, ...] * dt
```

随后交给：

```python
env.exec_actions(env_action, action_timestamps)
```

而 `openarm_umi_env.py` 会在每个 `target_time` 前等待，到点后执行一条动作：

```python
for action, target_time in zip(actions, timestamps):
    wait_until(target_time)
    exec_action(action)
```

所以：

- 单条动作的下发节拍由 `frequency` 决定
- 一次推理覆盖多少条动作，由 `steps_per_inference` 决定


## 运行入口

常用入口是：

```bash
python scripts/eval/eval_openarm_umi.py \
  --input outputs/umi_hdf5_run1/checkpoints/latest.ckpt \
  --urdf-path src/kinematics/description/openarm/urdf/openarm_bimanual.urdf \
  --camera-device 1 \
  --side left \
  --robot-id my_openarm_umi_left2 \
  --port can1
```

运行前通常还需要先配置 CAN：

```bash
lerobot-setup-can --mode=setup --interfaces=can0,can1
```

## 代码阅读建议

如果要继续追具体实现，推荐按这个顺序看：

1. `eval_openarm_umi.py`
2. `openarm_umi_episode.py`
3. `openarm_umi_env.py`
4. `umi/real_world/real_inference_util.py`
5. `diffusion_policy/common/pose_repr_util.py`
6. `umi/common/pose_util.py`
