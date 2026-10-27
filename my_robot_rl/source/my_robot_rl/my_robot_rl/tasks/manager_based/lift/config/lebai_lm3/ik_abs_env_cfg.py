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

import my_robot_rl.tasks.manager_based.lift.mdp as mdp

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
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.18]),# 这里测量过Lebai的距离是0.18m,用vscode预览测量的
        )
        # 2. 🔥 同步修改传感器的偏移
        self.scene.ee_frame.target_frames[0].offset.pos = [0.0, 0.0, 0.18]

@configclass
class LebaiCubeLiftEnvCfg_PLAY(LebaiCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


# ##这里先不用
# # Deformable object lift environment.
# ##


# @configclass
# class LebaiTeddyBearLiftEnvCfg(LebaiCubeLiftEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         self.scene.object = DeformableObjectCfg(
#             prim_path="{ENV_REGEX_NS}/Object",
#             init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.05), rot=(0.707, 0, 0, 0.707)),
#             spawn=UsdFileCfg(
#                 usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
#                 scale=(0.01, 0.01, 0.01),
#             ),
#         )

#         # Make the end effector less stiff to not hurt the poor teddy bear
#         self.scene.robot.actuators["panda_hand"].effort_limit_sim = 50.0
#         self.scene.robot.actuators["panda_hand"].stiffness = 40.0
#         self.scene.robot.actuators["panda_hand"].damping = 10.0

#         # Disable replicate physics as it doesn't work for deformable objects
#         # FIXME: This should be fixed by the PhysX replication system.
#         self.scene.replicate_physics = False

#         # Set events for the specific object type (deformable cube)
#         self.events.reset_object_position = EventTerm(
#             func=mdp.reset_nodal_state_uniform,
#             mode="reset",
#             params={
#                 "position_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
#                 "velocity_range": {},
#                 "asset_cfg": SceneEntityCfg("object"),
#             },
#         )

#         # Remove all the terms for the state machine demo
#         # TODO: Computing the root pose of deformable object from nodal positions is expensive.
#         #       We need to fix that part before enabling these terms for the training.
#         self.terminations.object_dropping = None
#         self.rewards.reaching_object = None
#         self.rewards.lifting_object = None
#         self.rewards.object_goal_tracking = None
#         self.rewards.object_goal_tracking_fine_grained = None
#         self.observations.policy.object_position = None
