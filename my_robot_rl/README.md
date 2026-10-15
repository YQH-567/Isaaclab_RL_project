
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

### 3.2 机器人 Asset 与物理模型验证
首先明白哪些配置应该在哪设置呢？：
| 参数类别 | 推荐存放位置 | 举例 |
|---|---|---|
| 机器人几何结构 | USD | Link、Visual Mesh、Collision Mesh |
| 关节物理结构 | USD | Joint Type、Axis、Body0/Body1、关节局部坐标系 |
| 质量与惯量 | USD | Mass、Center of Mass、Inertia |
| 机械关节限位 | 优先 USD | Lower/Upper Joint Limit |
| Articulation Root API | USD | 哪个节点是 Articulation Root |
| 基础物理求解参数 | USD 保存基准值，Python 可覆盖 | Solver Iterations、Sleep Threshold |
| 驱动与控制模型 | lm3.py | ImplicitActuatorCfg、PD 参数 |
| 初始关节姿态 | lm3.py | init_state.joint_pos |
| 任务中的机器人位置 | 环境 SceneCfg | prim_path、环境中的相对位置 |
| 强化学习动作空间 | 环境 ActionsCfg | JointPositionActionCfg |
| 奖励和终止条件 | 环境 RewardsCfg / TerminationsCfg | 距离奖励、成功判定 |
| 域随机化 | 环境 EventCfg | 质量、摩擦、初始状态随机化 |
#### 1. Articulation Root 检查
URDF从solidworks导出并且按照3.1放在对应的位置，（预览拓展能打开查看验证的话那路径什么的应该没有很大问题）,然后启动空模板`./isaaclab.sh -s`,然后左上角点击`importer`导入URDF文件，记得选固定,看图的选择：

<p>
<img src="./docx/isaaclab的URDF导入选择小tips.png" width=500>
</p>

- ① 点击 Play
先让 PhysX 初始化机械臂物理系统。


- ② 打开 Physics Inspector
找到 LM3 对应的 Articulation 并刷新。


- ③ 检查六个自由度
应当能够看到 joint_1 到 joint_6 六个可动关节，并读取相应关节状态。
这里还需要注意的是。如果Physics Inspector没有刷新出来，看看你是不是有root_joint,root_joint有没有`Articulation Root`,如下图，特别注意右边的参数

<p>
<img src="./docx/isaac中的Physics Inspector检查.png" width=600>
</p>


#### 2. 六个关节参数调节及其依据

- 参数调节：

1. 理论依据:[自动控制原理调节依据](./docx/机器人关节PD控制器调参与验证.md)

2. isaaclab的增益调节器网址：https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup/ext_isaacsim_robot_setup_gain_tuner.html

- isaac的增益调节器如何使用：

    在左侧 Gain Tuner 的 Test Gains Settings：
1. 将 Sinewave 切换为 Step Function。
2. 只勾选 joint_1 的 Test。
3. 设置 Step Minimum 和 Step Maximum。
4. 设置 Period、Test Duration。
5. 点击 RUN TEST。

- 以joint_1的阶跃响应的Kp和Kd的参数调节为例子：

    以下是不同Kd的阶跃响应为例子，数值为1-5，当Kd为1时，系统明显处于欠阻尼的状态，峰值大于目标值且收敛，对于欠阻尼来说，有超调量，响应时间，调节时间等的这些概念，特别对于阻尼选择这个东西，你可以大致有以下的概念：

    <span style="color:red; background:yellow;">
    阻尼比在 0.6～0.8 左右确实是经典二阶控制系统中经常采用的设计参考，它可以在响应速度和超调之间取得比较好的折中。
    但机器人关节控制器没有“ζ = 0.7 一定最优”这回事。

    例如：
    - 如果你需要快速响应，允许少量超调，可能选择较小阻尼。
    - 如果你需要精确抓取，必须尽量避免超调，可能更倾向于临界阻尼附近。
    - 如果存在严格的速度、力矩限制，单纯按照 ζ 设计也不够。
    - 如果机械臂存在复杂的关节耦合，单关节二阶模型只是近似。
    </span>

- 以下是isaaclab的调节图，分别是kd=1和kd=5的，2和3和4我就不放了，放在了docx里面：

    <p>
    <img src="./docx/lebai_joint_1的Kd=1阶跃响应图.png" width=500>
    </p>

    <p>
    <img src="./docx/lebai_joint_1的Kd=5阶跃响应图.png" width=500>
    </p>

- 正弦响应调节 

调节参数选择及其结果，比如joint_6，其他结果也差不多，2轴有点相位滞后，其他轴图片放docx里面：

<p>
<img src="./docx/isaaclab_joint_6的正弦响应图.png" width=500>
</p>

- 验证全部关节的流程：

    | 阶段 | 你需要完成的内容 |
    |---|---|
    | 1 | 六个关节分别完成小幅阶跃测试 |
    | 2 | 记录 Kp、Kd、超调量、调节时间和峰值速度 |
    | 3 | 每个关节选一组稳定且响应合理的参数 |
    | 4 | 验证两三个不同的机械臂安全姿态 |
    | 5 | 保存 USD 的基础 Drive 配置 |
    | 6 | 编写 lebai_lm3.py 和 Isaac Lab 控制测试 |
    | 7 | 完成多关节运动、Reset 和稳定性验证 |

或者换一种简单流程图：

            1. 静态检查
                |
                v
            2. 阶跃响应
                |
                v
            3. 正弦响应
                |
                v
            4. 多关节运动
                |
                v
            5. 不同负载

- 总结来说：

阶跃响应看什么？
1. 稳态误差
2. 超调
3. 调节时间
4. 振荡 (突然给目标，系统是否稳定)

正弦响应测试什么？
1. 相位滞后

    例如：
    目标：

    $$q_d=10\sin(t)$$

    实际：
    $$q=10\sin(t-30^\circ)$$
    说明：
    响应慢。

2. 幅值衰减

    目标：10°
    实际：8°
    说明：
    高频跟踪能力不足。

3. 高频稳定性

    机器人运动不是只有阶跃。
    RL里面：
    action不断变化：
    类似：
```python
0.1s:
目标1

0.2s:
目标2

0.3s:
目标3
```

所以动态跟踪能力重要。


### 3.3 ManagerBased 环境构建
    SceneCfg、ActionsCfg、ObservationsCfg、RewardsCfg、TerminationsCfg、EventCfg，重点解释各部分的数据 shape 和交互

### 3.4 PPO YAML 与环境的连接
    Gym 注册 → 环境配置 → skrl agent 配置 → train.py → rollout → PPO 更新 → checkpoint。
    解释 Actor、Critic、GAE、clip 和学习率，结合 YAML 给出实际配置。

### 3.5 实验结果与诊断
    TensorBoard、末端到目标点距离、成功率、episode length、训练配置与 checkpoint 回放。

### 3.6 抓取与 Sim2Real 扩展
    在 Reach 稳定以后加入抓取任务，再根据进度增加随机化与 Residual RL。