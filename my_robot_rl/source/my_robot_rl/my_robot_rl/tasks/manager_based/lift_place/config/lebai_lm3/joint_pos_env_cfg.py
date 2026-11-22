# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

#from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.schemas.schemas_cfg import (
    RigidBodyPropertiesCfg,
    ArticulationRootPropertiesCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from my_robot_rl.tasks.manager_based.lift_place import mdp
from my_robot_rl.tasks.manager_based.lift_place.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from my_robot_rl.assets.lebai_lm3_lmg90 import LEBAI_LM3_LMG90_CFG  # isort: skip

import math
import isaaclab.sim as sim_utils


def make_nut_cfg(name, pos, yaw=0.0, display_only=False):
    """Create an M16 nut for scene visualization."""

    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos,
            rot=[
                math.cos(yaw / 2),
                0.0,
                0.0,
                math.sin(yaw / 2),
            ],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_nut_m16.usd",

            articulation_props=ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            scale=(1.0, 1.0, 1.5),#修改规格，长宽高

            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=display_only,
                kinematic_enabled=display_only,
            ),

            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.72, 0.74, 0.78),
                metallic=0.85,
                roughness=0.28,
            ),
        ),
    )



@configclass
class LebaiCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Lebai as robot
        self.scene.robot = LEBAI_LM3_LMG90_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (Lebai)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint_[1-6]"], scale=0.5, use_default_offset=True
        )

        # #这是用原本的manager-based的gripper动作配置，如果用ik的话什么的需要换回来这个版本
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_r_joint1"],
            open_command_expr={"gripper_r_joint1": 1.0},  #弧度制
            close_command_expr={"gripper_r_joint1": 0.0},
        )

        
        
        
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "gripper_base_link"

        # # Set Cube as object  设置小方块
        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, 0.055], rot=[1, 0, 0, 0]),
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        #         scale=(0.8, 0.8, 0.8),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )

        # # Set M16 Nut as object
        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=RigidObjectCfg.InitialStateCfg(
        #         pos=[0.4, 0.0, 0.055],
        #         rot=[1.0, 0.0, 0.0, 0.0],
        #     ),
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_nut_m16.usd",
        #         scale=(1.0, 1.0, 1.5),#修改规格，长宽高

        #         # Disable the articulation root in the original USD
        #         articulation_props=ArticulationRootPropertiesCfg(
        #             articulation_enabled=False,
        #         ),

        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #         # visual_material=sim_utils.PreviewSurfaceCfg(
        #         #     diffuse_color=(0.75, 0.75, 0.78),
        #         # ),
        #     ),
        # )


        # Main target nut: dynamically simulated
        self.scene.object = make_nut_cfg(
            "Object",
            [0.40, 0.0, 0.055],
            yaw=0.0,
            display_only=False,
        )

        # # Additional nuts: visualization only
        # self.scene.nut_01 = make_nut_cfg(
        #     "Nut_01", [0.32, -0.10, 0.055], yaw=-0.4,# display_only=True
        # )

        # self.scene.nut_02 = make_nut_cfg(
        #     "Nut_02", [0.40, -0.10, 0.055], yaw=0.3, #display_only=True
        # )

        # self.scene.nut_03 = make_nut_cfg(
        #     "Nut_03", [0.48, -0.10, 0.055], yaw=0.8, #display_only=True
        # )

        # self.scene.nut_04 = make_nut_cfg(
        #     "Nut_04", [0.32, 0.10, 0.055], yaw=1.1, #display_only=True
        # )

        # self.scene.nut_05 = make_nut_cfg(
        #     "Nut_05", [0.48, 0.10, 0.055], yaw=-0.7, #display_only=True
        # )


        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/lebai_lm3/base_link",   #这两个link不是很确定
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/lmg90_gripper_inertial_stable/gripper/gripper_base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.21],
                    ),
                ),
            ],
        )


# @configclass
# class LebaiCubeLiftEnvCfg_PLAY(LebaiCubeLiftEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()
#         # make a smaller scene for play
#         self.scene.num_envs = 50
#         self.scene.env_spacing = 2.5
#         # disable randomization for play
#         self.observations.policy.enable_corruption = False
