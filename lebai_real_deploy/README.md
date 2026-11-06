# 第四章节 LM3 实机部署工程结构与开发流程

## 1. 工程目录

```text
        isaac_space_01/
        ├── my_robot_rl/                  # 训练端：Isaac Lab + SKRL
        │   ├── assets/                   # LM3 的 USD、URDF 等仿真资产
        │   ├── tasks/                    # Reach 任务、观测、奖励、终止条件
        │   ├── scripts/                  # 训练、Play、评估、模型导出
        │   └── logs/                     # 训练日志与 checkpoint
        │
        ├── g1_learning/                  # G1 相关学习与实验
        │
        └── lm3_real_deploy/              # 部署端：不依赖 Isaac Lab
            ├── configs/
            │   ├── robot.yaml            # 机器人通信与控制参数
            │   └── policy.yaml           # 模型、观测、动作配置
            ├── checkpoints/
            │   └── lm3_reach_policy.pt   # 导出的 TorchScript Actor
            ├── src/
            │   ├── robot/
            │   │   └── lebai_client.py   # 封装乐白 SDK
            │   ├── policy/
            │   │   └── actor.py          # 加载模型、执行推理
            │   └── control/
            │       ├── adapter.py        # 实机状态与策略输入输出之间的转换
            │       └── safety.py         # 部署前的安全检查
            ├── scripts/
            │   ├── test_connection.py    # 测试通信与状态读取
            │   ├── test_policy_offline.py# 不连接机器人，验证模型
            │   ├── test_joint_control.py # 独立验证关节控制接口
            │   └── deploy_reach.py       # Reach 部署主程序
            ├── docs/
            │   └── real_deployment.md    # 部署步骤、实验记录、已知问题
            ├── requirements.txt          # Python 运行依赖
            ├── .gitignore                # 忽略日志、缓存、敏感配置等
            └── README.md                 # 项目入口与使用说明
```

## 2. 各文件的职责

### 2.1 配置文件：`configs/`

- `robot.yaml`：保存机器人 IP、通信参数、控制频率、超时配置等。不得把密码或其他敏感凭据提交到公开仓库。
- `policy.yaml`：保存模型路径、Observation 维度及字段顺序、动作缩放、动作限制和目标到达阈值等。

配置文件用于集中管理参数，避免在多个 Python 文件中重复硬编码。

### 2.2 机器人通信：`src/robot/lebai_client.py`

职责：

- 初始化乐白 SDK 并连接 LM3。
- 读取真实关节位置、关节速度及机器人状态。
- 封装经过验证的关节目标下发接口。
- 处理通信异常、超时和连接断开。

该文件不负责强化学习推理，也不负责决定策略应该采取什么动作。

### 2.3 策略推理：`src/policy/actor.py`

职责：

- 使用 PyTorch 加载导出的 TorchScript 模型。
- 将 Observation 转换为模型需要的张量格式。
- 在推理模式下计算 Action。
- 检查输出形状和数值是否有效。

部署端不需要重新构建 SKRL PPO Agent，也不需要加载优化器和 Critic 网络。

### 2.4 观测与动作适配：`src/control/adapter.py`

这是 Sim2Real 中非常关键的模块。

职责：

- 将真实机器人状态转换成训练时使用的 25 维 Observation。
- 保持字段顺序、单位、坐标系和归一化方式一致。
- 将策略输出转换为训练环境定义的动作含义。
- 根据训练时的动作定义，生成对应的关节目标。

例如，如果训练时的动作是相对关节目标的偏移量，就不能把它直接当作绝对关节角发送。

**必须从训练环境的 Action 定义中确认动作含义，不能凭模型输出数值猜测。**

### 2.5 软件安全检查：`src/control/safety.py`

职责：

- 检查 Observation 和 Action 是否包含 NaN 或 Inf。
- 检查关节目标是否越界。
- 检查允许的速度、动作变化量和工作空间约束。
- 处理超时、状态异常和策略输出异常。
- 在异常情况下进入明确的停止或退出流程。

这些软件检查不能替代机器人控制器自身的安全功能、硬件急停和实验室操作规范。

### 2.6 测试脚本：`scripts/`

- `test_connection.py`：验证 SDK 安装、网络连接和状态读取。
- `test_policy_offline.py`：不连接机器人，验证模型加载、输入输出形状及数值有效性。
- `test_joint_control.py`：独立验证关节控制接口。实际运动必须采用受控的低速测试。
- `deploy_reach.py`：最终部署入口，负责协调状态读取、观测构造、策略推理、安全检查和控制下发。

测试脚本应当能够独立运行，不能要求每次排查通信问题都启动完整的强化学习系统。

### 2.7 文档与依赖

- `docs/real_deployment.md`：记录安装步骤、部署流程、实验结果、故障排查和已知限制。
- `requirements.txt`：记录部署端通用 Python 依赖。
- `.gitignore`：排除虚拟环境、缓存、临时日志和本地敏感配置。
- `README.md`：提供项目介绍、安装入口、测试顺序和使用说明。

## 3. 部署依赖的选择原则

部署环境只安装运行代码实际需要的包，不复制整个 Isaac Lab 训练环境。

| 依赖 | 用途 |
|---|---|
| PyTorch | 加载 TorchScript 并执行策略推理 |
| NumPy | 数值处理和状态转换 |
| PyYAML | 读取 YAML 配置 |
| `lebai-sdk` | 使用现有的同步乐白 SDK 接口 |
| Python 标准库 | 文件、日志、异常处理、时间和信号处理等 |

以下组件通常不需要安装到部署环境：

- Isaac Lab、Isaac Sim
- SKRL
- Stable-Baselines3
- TensorBoard、W&B

前提是部署代码不直接导入这些训练或监控组件。

## 4. requirements.txt 初始版本

```text
numpy
PyYAML
lebai-sdk
```

PyTorch 暂时单独安装，以明确选择 CPU 版本，而不是意外安装带 CUDA 依赖的版本。

完成独立环境测试后，应将通用依赖固定到经过验证的版本，并记录 Python 与 PyTorch 版本。不要直接把训练环境完整的 `pip freeze` 输出复制过来。

## 5. 部署环境安装

以 Ubuntu 24.04 和 Python 3.11 为例：

```bash
conda create -n lm3_deploy python=3.11 -y
conda activate lm3_deploy

python -m pip install torch==2.7.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 安装通用运行依赖
cd ~/isaac_space_01/lm3_real_deploy
python -m pip install -r requirements.txt
python -m pip check
```

检查依赖：

```bash
python -m pip show torch numpy PyYAML lebai-sdk
python -m pip check
```



随后分别测试模型加载、SDK 导入和真实机器人连接。

- 模型加载，在python环境里测试，笔记本和台式机都要测：

```python
import torch

policy = torch.jit.load(
    "checkpoints/lm3_reach_policy.pt",
    map_location="cpu",
)
policy.eval()

obs = torch.zeros((1, 25), dtype=torch.float32)

with torch.inference_mode():
    action = policy(obs)

print("Action:", action)
```
输出：

```bash
Action: tensor([[ 0.1723, -1.0820,  4.7594, -1.7930, -0.9335, -1.6624]])
# 但这里需要注意 这些是策略动作，不是关节角，也不是可以直接发送给 LM3 的命令。
# 且这里的结果需要跟env_isaaclab训练的输出一致：
训练环境：
[ 0.1723, -1.0820, 4.7594, -1.7930, -0.9335, -1.6624 ]

部署环境：
[ 0.1723, -1.0820, 4.7594, -1.7930, -0.9335, -1.6624 ]
```
## 6. 逐步测试和怎么写代码流程

## 核心思路与理解 ##

之前的 Reach 任务使用的是：
$$o_t=[q_{\mathrm{rel}},\dot q_{\mathrm{rel}},p_{\mathrm{cmd}},Q_{\mathrm{cmd}},a_{t-1}]$$

共 25 维：

| 索引 | 内容 | 维度 |
|---|---|---|
| 0:6 | 相对默认位置的六轴关节角 | 6 |
| 6:12 | 相对默认速度的六轴关节速度 | 6 |
| 12:19 | 目标位置与四元数，根坐标系下 | 7 |
| 19:25 | 上一次策略动作 | 6 |

特别注意，pose_command 是训练环境的命令生成器提供的目标，并不是直接使用 SDK 返回的 TCP 位姿。

- Observation Adapter：
```bash
        Lebai SDK 实机状态
            +
        目标位姿
            +
        上一时刻动作
            ↓
        25 维 Observation
            ↓
        TorchScript Actor
```
- Action Adapter：
```bash
    Actor 输出 6 维 Action
        ↓
    按 JointPositionActionCfg 缩放
        ↓
    默认关节位置偏移
        ↓
    关节目标位置
        ↓
    独立安全检查
```
这里必须保留训练时用的：
```python
    scale = 0.5
    use_default_offset = True
```

## 实现步骤 ##

### 1. 实机链接测试`scripts/test_connection.py`,实现 SDK 只读状态测试 ###
### 2. 进入 `test_policy_offline.py` 阶段 ###

这个阶段主要完成三件事：
- 从 checkpoints/ 加载 TorchScript 策略。
- 输入 25 维测试 Observation，检查是否得到六维 Action。
- 检查模型输出是否包含 NaN、Inf，并验证多次推理结果一致。
### 3. 编写`adapter.py`文件测试 ###
终于到 Sim2Real 的核心部分了。
我们先实现最简单、也最容易验证的两部分：
-  真实关节位置转换为 PPO Observation 中的 joint_pos_rel。
-  PPO Action 转换为 Isaac Lab 定义的目标关节位置。

这里需要理解两个公式

Observation：
$$ q_{\mathrm{rel}}=q_{\mathrm{real}}-q_{\mathrm{default}}$$
它让神经网络知道当前关节相对于训练默认姿态偏离了多少。

Action：
$$ q_{\mathrm{target}}=q_{\mathrm{default}}+0.5a$$
注意，a 是策略输出，不是目标关节角。代码当前仅适用于你训练时配置的 `JointPositionActionCfg(scale=0.5, use_default_offset=True)`，后续还要确认动作裁剪、关节名称顺序以及真实机器人运动限制。

### 4. 再写一个不控制机械臂的测试保存到 `scripts/test_adapter.py` ###

运行`python scripts/test_adapter.py`,预期 q_rel 接近零，零动作转换得到默认关节角



注意：依赖清单必须经过独立环境验证。仅仅安装成功，不代表模型输入、机器人接口和动作定义已经匹配。

## 7. 推荐的开发与验证顺序

1. 在台式机创建独立部署环境，不修改 Isaac Lab 的训练环境。
2. 安装 CPU 版 PyTorch 和部署依赖。
3. 验证 TorchScript 可以独立加载和推理。
4. 验证策略输入的 25 维结构、归一化和动作映射。
5. 将代码提交 GitHub，在笔记本克隆同一仓库。
6. 在实验室网络环境中测试 SDK 连接和状态读取。
7. 独立验证关节控制接口与停止行为。
8. 完成安全检查后，再开展低速、受控的策略部署。
9. 记录实机成功率、到达时间、位置误差、失败案例及仿真与实机差异。

每一步通过后再进入下一步。模型能输出动作，不等于机器人已经可以安全执行该动作。






    flowchart TD
        A["生成目标轨迹 q_des(t)"] --> B["记录命令发送时间"]
        B --> C["发送机器人控制命令"]
        C --> D["读取机器人关节反馈"]
        D --> E["记录反馈接收时间"]
        E --> F["统一时间基准"]
        F --> G["检查关节顺序、单位、零点"]
        G --> H["按时间戳对齐目标与反馈"]
        H --> I["计算关节误差"]
        I --> J["绘制每个关节误差曲线"]
        J --> K["分析延迟、偏置、跟踪误差"]

