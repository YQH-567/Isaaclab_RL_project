# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from my_robot_rl.tasks.manager_based.lift import mdp
from my_robot_rl.tasks.manager_based.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from my_robot_rl.assets.lebai_lm3_lmg90 import LEBAI_LM3_LMG90_CFG  # isort: skip

from isaaclab.sensors import ContactSensorCfg

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

        # 这是用原本的manager-based的gripper动作配置，如果用ik的话什么的需要换回来这个版本
        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["gripper_r_joint1"],
        #     open_command_expr={"gripper_r_joint1": 1.0},  #弧度制
        #     close_command_expr={"gripper_r_joint1": 0.0},
        # )

        # 这是用乐白新加的带距离门控的gripper动作配置
        self.actions.gripper_action = mdp.GatedGripperActionCfg(
            asset_name="robot",
            joint_names=["gripper_r_joint1"],
            open_command_expr={"gripper_r_joint1": 1.0},
            close_command_expr={"gripper_r_joint1": 0.0},
            close_distance=0.02,
        )
        
        
        
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "gripper_base_link"

        #启用机器人刚体触觉：
        self.scene.robot.spawn.activate_contact_sensors = True

        # Set Cube as object  设置小方块
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/lebai_lm3/base_link",   #这两个link不是很确定
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/lmg90_gripper_inertial_stable/gripper/gripper_base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.18],
                    ),
                ),
            ],
        )


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
