# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

# import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
# from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

import my_robot_rl.tasks.manager_based.reach.mdp as mdp
from my_robot_rl.tasks.manager_based.reach.reach_env_cfg import ReachEnvCfg
##
# Pre-defined configs
##
from my_robot_rl.assets.lebai_lm3 import LEBAI_LM3_CFG  # isort: skip


##
# 环境配置
##


@configclass
class lebai_lm3ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):   
        # 执行父类的后初始化
        """
        IsaacLab 的环境配置（EnvCfg）通常是多层继承的。
        父类配置（如 LiftEnvCfg）会设定许多基础的默认值（机器人、场景、观察、动作空间等）。
        当子类（如 FrankaCubeLiftEnvCfg）想要基于这些基础进行修改时，必须先执行父类的 __post_init__，
        将父类的所有默认配置加载到当前实例中，否则子类将在一个“空”的配置基础上进行修改，导致大量配置缺失。
        @configclass 已经处理了配置字段的默认实例。父类 __post_init__() 主要负责执行额外的初始化逻辑。
        你这里必须调用它，是因为 ReachEnvCfg.__post_init__() 设置了：
        self.decimation = 2这些参数
        """
        super().__post_init__()

        # 把机器人换成lebai_lm3
        self.scene.robot = LEBAI_LM3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 覆盖的奖励，这里可以对照着reach_env_cfg.py的class RewardsCfg看，看它任务设置
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link_6"]  #还没放加爪，那我就先用6轴末端吧。原来的panda_hand就是panda_finger_joint.*
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link_6"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link_6"]

   

        #我在这里说明通用的这个reach_env_cfg.py设置的主要奖励和惩罚：
        """
        任务奖励或惩罚：
        1. 当前位置和目标位置的距离（在世界坐标系中），计算方式是欧式距离计算，这里设置weight=-0.2，那就是越远惩罚越高
        2. 双曲正切函数，范围在[-1,1]之间，防止梯度爆炸和奖励绑架，准确严谨说是将位置误差转换成一个有界的跟踪奖励，这样不会让任务1因为距离目标点无限远而导致惩罚无限大，牢牢锁定在-1到1之间，weight=0.1，如果无限远那奖励就是1-1=0
        3. 约束机械臂末端旋转角和旋转姿态，偏差大就惩罚，因为设置的weight=-0.1，如果学不会转身可以把这里的weight改大
        动作惩罚：
        1. 欧几里德距离动作变化率惩罚，weight=-0.0001，防止动作抖动也保护电机
        2. 关节速度（Joint Velocity）的欧几里得惩罚项，Penalty=v1^2​+v2^2​+...+v6^2​,不过这里weight=-0.0001，温和引导作用，不然一下子惩罚太大，比奖励多得多那机器人可能都不走了，原地不动
        panda_joint.*
        """
        
        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint_.*"], scale=0.5, use_default_offset=True
        )  #这个arm_action的定义是arm_action: ActionTerm = MISSING，所以这里必须写明白self.actions.arm_action =什么
        # 重写命令生成器主体
        # 末端执行器沿 z 方向
        self.commands.ee_pose.body_name = "link_6"  #这里说明末端执行器是joint_6,因为我暂时不设置加爪，并且设置夹爪配置一般在 my_robot_rl/source/my_robot_rl/my_robot_rl/assets/lebai_lm3.py这里
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)#这个的意思是设置为固定值，而不是俯仰角姿态随机，这里RPY中只有这个pitch是一定要自己设置的，因为是MISSING

        print(f"chushiweizhi: {self.scene.robot.data.default_joint_pos[0]}" )
        print(f"第一轴的初始位置： {self.scene.robot.data.joint_pos[0]}")
        #print(f"查看手臂输出action: {env.action_manager.get_term("arm_action").raw_actions[0]}")     

# @configclass
# class FrankaReachEnvCfg_PLAY(FrankaReachEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()
#         # make a smaller scene for play
#         self.scene.num_envs = 50
#         self.scene.env_spacing = 2.5
#         # disable randomization for play
#         self.observations.policy.enable_corruption = False
