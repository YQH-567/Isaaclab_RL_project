# 机器人关节 PD 控制器调参与验证

> 适用对象：Isaac Sim 5.1 / Isaac Lab，六自由度 LM3 机械臂。本文可作为 `my_robot_rl/docx/` 中的技术学习笔记。数值和曲线必须以实际测试为准，不应把示例值作为实机标定结果。

## 1. 调参的目标与前置条件

控制器调参不是把机械臂调到“看起来不抖”即可，而是验证关节在已知目标下具有**稳定、快速、准确、可重复**的响应，同时不违反速度、力矩和关节位置限制。

调参前应先确认：

1. USD 只有正确的 Articulation Root，固定底座连接正确，六个关节均能被 Physics Inspector 识别。
2. URDF/USD 中的关节轴、局部坐标系、几何、碰撞、质量和惯量基本合理。
3. 核查关节位置、速度和力矩限值的来源，区分厂家规格、CAD 数据、导入默认值和仿真试验值。
4. 采用固定的物理时间步、重力和初始姿态；先单轴、空载测试，随后扩展到不同姿态及负载。
5. 测试场景能停止、复位，并确认没有尚未处理的持续仿真错误。

**重要区分：**
URDF `<dynamics damping="..." friction="..."/>` 表示关节的被动动力学参数；USD Joint Drive 或 Isaac Lab `ImplicitActuatorCfg` 的 `stiffness`、`damping` 表示驱动控制器增益。二者不应混为一谈。

## 2. PD 的控制原理

对单关节位置控制，可采用如下近似关系：

$$
\tau=K_p(q_d-q)+K_d(\dot q_d-\dot q)
$$

其中：

- $q_d$：目标关节位置；$q$：实际关节位置。
- $\dot q_d$：目标关节速度；$\dot q$：实际关节速度。
- $K_p$：位置增益（stiffness）；$K_d$：速度增益（damping）。
- $\tau$：驱动输出力矩，实际受到力矩上限与物理约束限制。

对于简化的单关节线性模型：

$$
J\ddot q+b\dot q=\tau
$$

若目标为固定位置、速度目标为零，令 $e=q_d-q$，得到：

$$
J\ddot e+(b+K_d)\dot e+K_p e=0
$$

由此得到自然频率与阻尼比的近似表达式：

$$
\omega_n=\sqrt{\frac{K_p}{J}},\qquad
\zeta=\frac{b+K_d}{2\sqrt{JK_p}}
$$

- 欠阻尼 $0<\zeta<1$：可能超调、振荡。
- 临界阻尼 $\zeta=1$：理想简化模型中较快且不振荡地收敛。
- 过阻尼 $\zeta>1$：通常不振荡，但可能响应较慢。

**适用边界：**
上述公式仅用于理解趋势。实际 LM3 为六轴非线性耦合机器人，惯量随姿态变化，存在重力、摩擦、关节限制、驱动饱和和隐式物理求解器；不能仅凭公式直接算出六轴最终增益。

## 3. 正规 PD 调参实验流程

### 3.1 建立基线和控制变量

记录各关节的 Drive Type、$K_p$、$K_d$、最大输出力矩、速度限制、起始位置、目标位置，以及 Physics dt、机器人构型和是否带负载。

在 LM3 的现有测试中，`Kp=100、Kd=1` 可作为一组**待验证的仿真基线**。这不是厂家指定参数，也不是六个关节的最终增益。若 USD 中部分关节原有增益不同，应先保存原始值。

一次实验只修改一个主要因素，保持其他条件一致。特别是不要同时改 $K_p$、$K_d$、Max Force、物理步长和机器人姿态，否则无法判断改善来自哪里。

### 3.2 阶跃响应测试（Step Function）

**目标：**
判断关节能否快速到达并稳定保持给定位置。

1. 复位机器人到已检查的安全姿态；选择 `joint_1`，其余关节保持不变。
2. 从小幅度目标开始，例如在安全区间内增加 $10^\circ$，不要先扫过全关节限位。
3. 设置足够长的测试持续时间，保证能够看到完整的上升与稳定过程。
4. 运行 Gain Tuner 的 Step Function 测试。
5. 记录目标位置 $q_d(t)$、实际位置 $q(t)$、速度 $\dot q(t)$；能获取时同时记录输出力矩 $\tau(t)$。
6. 回到相同初始条件重复测试，之后才修改 $K_p$ 或 $K_d$。

**观察：**

- 到达慢、明显滞后：检查增益偏低、输出力矩饱和、速度上限或目标变化过快。
- 明显超调、反复振荡：检查阻尼偏低、刚度过大、时间步和接触约束问题。
- 最终存在较大偏差：检查重力负载、力矩饱和、增益和被动摩擦等。
- 大幅关节姿态突变、连杆分离或碰撞异常：优先修复模型与约束，而不是继续提高增益。

### 3.3 阶跃响应评价指标

设阶跃幅值 $\Delta q=q_{\mathrm{final}}-q_{\mathrm{initial}}$：

**上升时间 $t_r$：**
通常定义为首次从阶跃变化量的 10% 到 90% 所用的时间（定义需在实验中保持一致）。

**超调量 $M_p$：**

$$
M_p=\frac{\max_t\left[s\left(q(t)-q_{\mathrm{final}}\right)\right]_+}{|\Delta q|}\times100\%,\quad s={sign}(\Delta q)
$$


其中 $[x]_+=\max(0,x)$。下降阶跃也应考虑方向，不能直接用“最大位置减目标位置”的公式。

**调节时间 $t_s$：**

从该时刻开始，位置始终保持在预设误差带内（例如阶跃幅值的 ±2%，也可以另选符合任务要求的绝对角度容差）。

**稳态误差：**

$$
e_{ss}=q_{\mathrm{final}}-\overline{q(t)}_{\mathrm{末段稳定窗口}}
$$

此外记录峰值速度、最大输出力矩、超限次数以及重复实验之间的波动。

**注意：**
如果测试始终无法进入误差带，不要硬填一个调节时间，应标记为“未收敛”。

### 3.4 正弦轨迹跟踪测试（Sinusoidal）

**目标：**
检查连续运动的跟踪能力、延迟和方向切换时的稳定性。

$$
q_d(t)=q_0+A\sin(2\pi f t)
$$

从安全的小振幅、低频率开始。比较目标与实际关节位置、速度，关注：

- 最大跟踪误差：$\max |q_d-q|$。
- 均方根跟踪误差：$\mathrm{RMSE}=\sqrt{\frac1N\sum_{k=1}^N(q_{d,k}-q_k)^2}$。
- 相位滞后、幅值衰减、速度峰值、力矩饱和。
- 频率增加时跟踪能力如何变化。

Gain Tuner 的位置测试通常将目标速度置为零以评估 Drive 的阻尼表现；这不等同于实机轨迹控制器完整的目标位置与前馈目标速度输入。因此正式控制链路还需验证实际使用的命令方式。

### 3.5 对照调参

建议先固定 $K_p$，逐步调整 $K_d$，观察超调与收敛变化；再固定一个合理阻尼范围，检查 $K_p$ 对响应速度和误差的影响。也可以先调整刚度使其能够到位，再提高阻尼抑制振荡。始终检查 Max Force 是否饱和。

| 组别 | Kp | Kd | 初始姿态 | 负载 | 上升时间 | 超调量 | 调节时间 | 稳态误差 | 备注 |
|---|---:|---:|---|---|---|---|---|---|---|
| A（基线） | 100 | 1 | 待记录 | 空载 | 待测 | 待测 | 待测 | 待测 | 非最终参数 |
| B | 待定 | 待定 | 同 A | 空载 | 待测 | 待测 | 待测 | 待测 | 每次控制变量 |
| C | 待定 | 待定 | 同 A | 空载 | 待测 | 待测 | 待测 | 待测 | 每次控制变量 |

表中的增益仅为当前仿真界面的记录值。比较 Isaac Sim GUI、USD 与 Isaac Lab Python 参数时，必须核对相应角度及增益单位。

### 3.6 多构型和负载验证

单关节基线通过后，按顺序测试：

1. 其余五个关节的单轴阶跃与正弦响应。
2. 收缩姿态与伸展姿态下的关节响应。
3. 多关节同时运动，观察耦合误差。
4. 加装 LMG90 后重新检查惯量、重力负载与控制响应。
5. 在安全范围内添加代表性抓取负载，检查稳定性及输出限制。
6. Reset 重复测试，确认仿真可复现。

只在单一姿态下“不抖动”不足以证明整机控制已通过验收。

## 4. Isaac Sim 5.1 Gain Tuner 操作

1. 打开 LM3 USD，确认 Articulation Root 及六个关节已经被 Physics Inspector 识别。
2. 打开 **Tools → Robotics → Asset Editors → Gain Tuner**。
3. 在上部关节列表检查模式 Position、驱动类型 Force，以及各轴 Stiffness / Damping。
4. 在 **Test Gains Settings** 只勾选 `joint_1` 的 Test，固定 Sequence，选择 **Step Function**。
5. 设置安全的阶跃最小与最大角度、周期和测试时长；不要直接使用整个 ±180° 的范围。
6. 运行测试并等待完成。若按钮仍为 `CANCEL TEST`，表示测试可能还在执行中。
7. 向下滚动并展开 **Charts**；需要时扩大 Gain Tuner 面板，在曲线列表中选中 `joint_1`。
8. 检查目标/实际位置、速度曲线。保存曲线截图与参数记录，然后改变单个增益重新试验。
9. 确认结果后再保存增益到 Physics Layer，并重新打开 USD 复验。

**截图排障提示：**
如果 Charts 标题可见但下方内容不可见，先检查测试是否结束以及窗口是否需要滚动/拉高。若仍空白，停止仿真、重新打开 USD 并重新初始化 Gain Tuner，查看 Console 是否有物理视图失效等错误。

## 5. 迁移到 Isaac Lab

推荐文件分工：

- `assets/robots/lm3/urdf/`：原始及修正后的结构模型。
- `assets/robots/lm3/usd/`：可用的 PhysX 物理资产及其默认驱动配置。
- `source/my_robot_rl/my_robot_rl/assets/lebai_lm3.py`：`ArticulationCfg`、初始状态及 `ImplicitActuatorCfg`。
- `scripts/tests/test_joint_response.py`：自动关节目标测试、采样记录、导出 CSV 和生成曲线。

示意读取方式：

```python
# 在已经初始化的 Isaac Lab 仿真循环中
joint_pos = robot.data.joint_pos       # (num_envs, num_joints)
joint_vel = robot.data.joint_vel       # (num_envs, num_joints)

# 记录位置和速度需要在每次 sim.step()、robot.update(dt) 后采样。
# 驱动力矩应使用当前 actuator / articulation 实际支持的力矩数据字段，
# 不要把简单的理论 PD 计算直接冒充实测 PhysX 力矩。
```

CSV 建议记录：`time, joint_name, q_target_rad, q_actual_rad, qd_target_rad_s, qd_actual_rad_s, effort_Nm, Kp, Kd, pose_id, load_id`。没有可信的力矩反馈时，该列保持为空并明确说明。

**进入强化学习的最低标准：**

六轴可稳定保持安全姿态、阶跃目标能够收敛、没有持续显著振荡、速度与输出限制未明显违规、Reset 可重复，多姿态下表现正常。无需在 Reach PPO 之前完成工业级实机伺服标定，但不能用训练奖励掩盖控制系统本身的问题。

## 6. 工程总结与面试表述

> 在仿真中，我先检查机器人的关节轴、限位、质量惯量及碰撞模型，再使用 Isaac Sim Gain Tuner 进行关节 PD 增益的阶跃响应和正弦跟踪实验。通过位置、速度及可用力矩数据评估上升时间、超调量、调节时间、稳态误差与饱和情况。控制变量调整 Kp/Kd，并在不同姿态和负载下验证；最终将经测试的参数整理进 Isaac Lab 的 ArticulationCfg 和自动化关节测试。实机上则区分底层伺服闭环与上层位置指令，不能直接将仿真 Drive 增益搬到真实控制器。

## 7. 参考资料

- NVIDIA Isaac Sim 5.1 — [Gain Tuner Extension](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup/ext_isaacsim_robot_setup_gain_tuner.html)
- NVIDIA Isaac Sim — [Joint Drive Gain Tuning Tutorial](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup_tutorials/joint_tuning.html)（注意不同版本的 UI 可能不同）
- Isaac Lab — [Writing an Asset Configuration](https://isaac-sim.github.io/IsaacLab/main/source/how-to/write_articulation_cfg.html)
