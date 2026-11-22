# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import DeformableObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import my_robot_rl.tasks.manager_based.lift_place.mdp as mdp

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from my_robot_rl.assets.lebai_lm3_lmg90 import LEBAI_LM3_LMG90_CFG  # isort: skip


##
# Rigid object lift environment.
##


@configclass
class LebaiCubeLiftEnvCfg(joint_pos_env_cfg.LebaiCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Lebai as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.  #我这没改PD，看看会不会出事
        self.scene.robot = LEBAI_LM3_LMG90_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (Lebai)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"],
            body_name="gripper_base_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose", 
                use_relative_mode=False,   #绝对模型下不能使用scale
                ik_method="dls",
                ik_params={"lambda_val": 0.01, "k_val": 1.0}),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.21]),# 这里测量过Lebai的距离是0.18m,用vscode预览测量的
        )
        # 2. 🔥 同步修改传感器的偏移
        self.scene.ee_frame.target_frames[0].offset.pos = [0.0, 0.0, 0.21]



