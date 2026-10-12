
## 3 LM3 机械臂 ManagerBased 强化学习项目
### 3.1 项目目标与系统结构
**1. 认清文件结构：**
```txt
保留现有的 .vscode、README.md、pyproject 等文件
my_robot_rl/
    assets/ 新增：机器人模型资产
        robots
            lm3/
                urdf/
                meshes/
                usd/
        grapper
            lmg90/

    scripts/
        tools/ 模型转换与检查
        tests/ 关节控制、复位测试
        skrl/ PPO 训练

    source/my_robot_rl/my_robot_rl/
        assets/ ArticulationCfg 配置代码
        tasks/ ManagerBased 环境
    docx/ 学习笔记、截图

    README.md
 ```
 **2. 模型本体urdf预览和isaaclab导入**
- meshes文件路径设置和插件预览：`package://robots/lm3/meshes/visual/base_link.stl`
安装好Vscode里的URDF-Preview可以进行机器人预览，查看关节转动范围/Visual/Collision/坐标系等等，如下图：

<p align="center">
<img src="./docx/vscode插件预览URDF.png" width=500>
<br>
<sub>vscode插件预览URDF</sub>
</p>

- 插件使用和配置文件路径编写

在当前的工作空间的.vscode/settings.json里面说明路径
```json
{
    "python-envs.defaultEnvManager": "ms-python.python:conda",
    "python-envs.defaultPackageManager": "ms-python.python:conda",
    "ros.distro": "jazzy",
    "python.analysis.extraPaths": [
        "./IsaacLab/source/isaaclab",
        "./IsaacLab/source/isaaclab_assets"
    ],
    "urdf-visualizer.packages": {
        "robots": "/home/l/isaac_space_01/my_robot_rl/assets/robots",
        "grapper": "/home/l/isaac_space_01/my_robot_rl/assets/grapper",
    }//核心是这里的路径配置，不然点击URDF预览是看不到的，同时自己理解一下这里的配置原理
}
```
 - 使用vscode的URDF
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