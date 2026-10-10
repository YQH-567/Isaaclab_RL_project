
## 3 LM3 机械臂 ManagerBased 强化学习项目
### 3.1 项目目标与系统结构
    LM3 + LMG90、Reach 任务、ManagerBasedRLEnv、skrl PPO。
    先说明为什么选 ManagerBased，以及这个任务如何评估成功。

### 3.2 机器人 Asset 与物理模型验证
    URDF/USD、本体坐标系、关节轴与限位、质量惯量、collision mesh、gripper mimic/drive、actuator、关节驱动参数。

### 3.3 ManagerBased 环境构建
    SceneCfg、ActionsCfg、ObservationsCfg、RewardsCfg、TerminationsCfg、EventCfg，重点解释各部分的数据 shape 和交互

### 3.4 PPO YAML 与环境的连接
    Gym 注册 → 环境配置 → skrl agent 配置 → train.py → rollout → PPO 更新 → checkpoint。
    解释 Actor、Critic、GAE、clip 和学习率，结合 YAML 给出实际配置。

### 3.5 实验结果与诊断
    TensorBoard、末端到目标点距离、成功率、episode length、训练配置与 checkpoint 回放。

### 3.6 抓取与 Sim2Real 扩展
    在 Reach 稳定以后加入抓取任务，再根据进度增加随机化与 Residual RL。