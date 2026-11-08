# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


from isaaclab.sensors import ContactSensorCfg


from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # 相当于加手指触觉传感器
    left_finger_contact = ContactSensorCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/lmg90_gripper_inertial_stable/"
            "gripper/gripper_l_link_finger"
        ),
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        debug_vis=False,
    )

    right_finger_contact = ContactSensorCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/lmg90_gripper_inertial_stable/"
            "gripper/gripper_r_link_finger"
        ),
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        debug_vis=False,
    )
##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            #pos_x=(0.2, 0.35), pos_y=(-0.25, 0.25), pos_z=(0.1, 0.35), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
            pos_x=(0.3, 0.3), pos_y=(-0.1, -0.1), pos_z=(0.2, 0.2), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)    #这里是抬起的目标点，目前我固定一个点
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:    #从这里看你的输入是几维
    """Observation specifications for the MDP."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)   #[2.12]
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)     #[2.12]
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)     #[2.3]
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})  #[2.7]
        actions = ObsTerm(func=mdp.last_action)  #[2.7]

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            #"pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP."""

#     reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
#     #原本weight为15.0
#     lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.10}, weight=15.0)
#     #原本weight为16.0
#     object_goal_tracking = RewTerm(
#         func=mdp.object_goal_distance,
#         params={"std": 0.3, "minimal_height": 0.10, "command_name": "object_pose"},
#         weight=16.0,
#     )
#     #原本weight为5.0
#     object_goal_tracking_fine_grained = RewTerm(
#         func=mdp.object_goal_distance,
#         params={"std": 0.05, "minimal_height": 0.10, "command_name": "object_pose"},
#         weight=5.0,
#     )

#     # action penalty  原本的weight为-1e-4
#     action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

#     joint_vel = RewTerm(
#         func=mdp.joint_vel_l2,
#         weight=-1e-4,
#         params={"asset_cfg": SceneEntityCfg("robot")},
#     )


# @configclass
# class RewardsCfg:
#     """Lebai LM3 + LMG90 grasp and lift rewards."""

#     # Actual TCP frame reaching
#     reaching_object = RewTerm(
#         func=mdp.lebai_tcp_object_distance,
#         params={"std": 0.08},
#         weight=3.0,
#     )  #原本是1.0

#     # Geometric enclosure
#     gripper_enclosure = RewTerm(
#         func=mdp.lebai_finger_enclosure,
#         params={
#             "left_tip_cfg": SceneEntityCfg(
#                 "robot", body_names="gripper_l_link_finger"
#             ),
#             "right_tip_cfg": SceneEntityCfg(
#                 "robot", body_names="gripper_r_link_finger"
#             ),
#         },
#         weight=2.0,
#     )

#     # Keep open while aligning
#     gripper_open_enclosure = RewTerm(
#         func=mdp.lebai_gripper_open_enclosure,
#         params={
#             "left_tip_cfg": SceneEntityCfg(
#                 "robot", body_names="gripper_l_link_finger"
#             ),
#             "right_tip_cfg": SceneEntityCfg(
#                 "robot", body_names="gripper_r_link_finger"
#             ),
#         },
#         weight=0.0,
#     )  #原本是0.2

#     # Close when properly enclosed
#     gripper_close_when_enclosed = RewTerm(
#         func=mdp.lebai_gripper_close_when_enclosed,
#         params={
#             "left_tip_cfg": SceneEntityCfg(
#                 "robot", body_names="gripper_l_link_finger"
#             ),
#             "right_tip_cfg": SceneEntityCfg(
#                 "robot", body_names="gripper_r_link_finger"
#             ),
#         },
#         weight=0.0,
#     )  #原本是0.8  0.0

#     # Do not close too early
#     premature_close = RewTerm(
#         func=mdp.lebai_premature_close,
#         params={
#             "left_tip_cfg": SceneEntityCfg(
#                 "robot", body_names="gripper_l_link_finger"
#             ),
#             "right_tip_cfg": SceneEntityCfg(
#                 "robot", body_names="gripper_r_link_finger"
#             ),
#         },
#         #weight=-0.8,
#         weight=0.0 #暂时取消这个惩罚，它可能在夹爪正确接触、闭合后依然产生负值
#     )

#     # Actual bilateral contact
#     grasp_contact = RewTerm(
#         func=mdp.lebai_bilateral_grasp,
#         weight=5.0,
#     )   #原本是4.0  

#     single_contact = RewTerm(
#         func=mdp.lebai_single_contact_penalty,
#         weight=0.0,
#     )# 原本是-0.2

#     # Continuous lift progress
#     lift_progress = RewTerm(
#         func=mdp.lebai_lift_progress,
#         params={
#             "rest_z": 0.055,
#             "target_lift": 0.10,
#         },
#         weight=6.0,
#     )  #原本是8.0

#     # Stable lifting success
#     lifting_object = RewTerm(
#         func=mdp.lebai_stable_lift,
#         params={
#             "rest_z": 0.055,
#             "min_lift": 0.04,
#             "max_object_speed": 0.15,
#         },
#         weight=120.0,
#     )  #原本是16.0

#     # Target tracking after grasp and lift
#     object_goal_tracking = RewTerm(
#         func=mdp.lebai_object_goal_distance,
#         params={
#             "std": 0.3,
#             "min_lift": 0.04,
#             "rest_z": 0.055,
#         },
#         weight=8.0,
#     )#原本是8.0,从来没有抓上过，先改0.0吧 0.0

#     object_goal_tracking_fine_grained = RewTerm(
#         func=mdp.lebai_object_goal_distance,
#         params={
#             "std": 0.05,
#             "min_lift": 0.04,
#             "rest_z": 0.055,
#         },
#         weight=2.0,
#     )#这个也是，原来是2.0,先改0.0吧  0.0

#     # Only regularize arm motion
#     action_rate = RewTerm(
#         func=mdp.lebai_arm_action_rate_l2,
#         weight=-1e-4,
#     )

#     joint_vel = RewTerm(
#         func=mdp.joint_vel_l2,
#         weight=-1e-4,
#         params={
#             "asset_cfg": SceneEntityCfg(
#                 "robot",
#                 joint_names=["joint_[1-6]"],
#             )
#         },
#     )

# #reward V3和V4的区别主要是rest_z的值从0.055改为0.021，max_object_speed从0.15改为10.0，lifting_object的weight从16.0改为8.0，object_goal_tracking的weight从8.0改为12.0，object_goal_tracking_fine_grained的weight从8.0改为0.0
# @configclass
# class RewardsCfg:
#     """LM3 + LMG90: Reach -> Grasp -> Lift."""

#     # 1. TCP approaching the cube
#     reaching_object = RewTerm(
#         func=mdp.lebai_tcp_object_distance,
#         params={"std": 0.15},
#         weight=3.0,
#     )

#     # 2. Finger geometry alignment
#     gripper_enclosure = RewTerm(
#         func=mdp.lebai_finger_enclosure,
#         params={
#             "left_tip_cfg": SceneEntityCfg(
#                 "robot",
#                 body_names=["gripper_l_link_finger"],
#             ),
#             "right_tip_cfg": SceneEntityCfg(
#                 "robot",
#                 body_names=["gripper_r_link_finger"],
#             ),
#         },
#         weight=2.0,
#     )

#     # 3. Real bilateral finger-object contact
#     grasp_contact = RewTerm(
#         func=mdp.lebai_bilateral_grasp,
#         params={"threshold": 0.5},
#         weight=5.0,
#     )

#     # 4. Continuous object lift
#     lift_progress = RewTerm(
#         func=mdp.lebai_lift_progress,
#         params={
#             "rest_z": 0.021,
#             "target_lift": 0.10,
#             "contact_threshold": 0.5,
#         },
#         weight=3.0,
#     )# 记录一下权重变化8.0 

#     # 5. Stable lifted object
#     lifting_object = RewTerm(
#         func=mdp.lebai_stable_lift,
#         params={
#             "rest_z": 0.021,
#             "min_lift": 0.05,
#             "max_object_speed": 10.0,
#             "contact_threshold": 0.5,
#         },
#         weight=8.0,#记录一下权重变化16.0
#     )
#     # 6. 开始增加抬升到目标点的奖励
#     object_goal_tracking = RewTerm(
#     func=mdp.lebai_goal_tracking,
#     params={
#         "command_name": "object_pose",
#         "std": 0.15,
#         "min_lift": 0.02,
#         "rest_z": 0.021,  # 按当前场景实测值修正
#     },
#     weight=12.0,
# )  #记录一下权重变化4.0

#     # 7.开始增加Target tracking after grasp and lift
#     object_goal_tracking_fine_grained = RewTerm(
#         func=mdp.lebai_object_goal_distance,
#         params={
#             "std": 0.3,
#             "min_lift": 0.04,
#             "rest_z": 0.021,
#         },
#         weight=0.0,
#     )#原本是8.0  0.0

#     # Arm action smoothness
#     action_rate = RewTerm(
#         func=mdp.lebai_arm_action_rate_l2,
#         weight=-1e-4,
#     )

#     # Arm joint velocity only
#     joint_vel = RewTerm(
#         func=mdp.joint_vel_l2,
#         params={
#             "asset_cfg": SceneEntityCfg(
#                 "robot",
#                 joint_names=["joint_[1-6]"],
#             ),
#         },
#         weight=-1e-4,
#     )


# reward V5版本和V6版本，这个V5其实跟没写是一样的，效果出来一模一样
@configclass
class RewardsCfg:
    """V5: reach -> bilateral grasp -> lift -> horizontal carry -> goal."""

    reaching_object = RewTerm(
        func=mdp.lebai_tcp_object_distance,
        params={"std": 0.15},
        weight=3.0,
    )

    gripper_enclosure = RewTerm(
        func=mdp.lebai_finger_enclosure,
        params={
            "left_tip_cfg": SceneEntityCfg("robot", body_names=["gripper_l_link_finger"]),
            "right_tip_cfg": SceneEntityCfg("robot", body_names=["gripper_r_link_finger"]),
        },
        weight=2.0,
    )

    grasp_contact = RewTerm(
        func=mdp.lebai_bilateral_grasp,
        params={"threshold": 0.5},
        weight=5.0,
    )

    # lift_progress = RewTerm(
    #     func=mdp.lebai_lift_progress,
    #     params={"rest_z": 0.021, "target_lift": 0.10, "contact_threshold": 0.5},
    #     weight=3.0,
    # )

    # lifting_object = RewTerm(
    #     func=mdp.lebai_stable_lift,
    #     params={
    #         "rest_z": 0.021,
    #         "min_lift": 0.05,
    #         "max_object_speed": 0.60,  # Candidate, not 10.0 m/s.
    #         "contact_threshold": 0.5,
    #     },
    #     weight=8.0,
    # )

    #这个是V5版本的，可以说这两个函数，计算函数很复杂，且，没有任何作用
    # transport_xy = RewTerm(   
    #     func=mdp.lebai_transport_xy,
    #     params={
    #         "command_name": "object_pose",
    #         "std": 0.20,
    #         "rest_z": 0.021,  # 按当前场景实测值修正
    #         "min_lift": 0.02,
    #     },
    #     weight=12.0,
    # )

    # goal_reached = RewTerm(
    #     func=mdp.lebai_goal_reached,
    #     params={
    #         "command_name": "object_pose",
    #         "goal_threshold": 0.05,
    #         "rest_z": 0.021,  # 按当前场景实测值修正
    #         "min_lift": 0.04,
    #         "max_object_speed": 0.20,
    #         "contact_threshold": 0.5,
    #         "max_tcp_distance": 0.08,
    #     },
    #     weight=30.0,
    # )

    #V6: reward for carrying cube toward the goal
    # v6_goal_reach = RewTerm(
    #     func=mdp.lebai_v6_goal_reach,
    #     params={"std": 0.20},
    #     weight=20.0,
    # )

    # V6: one-time reward when reaching the goal
    v6_goal_success = RewTerm(
        func=mdp.lebai_v6_success_bonus,
        params={"goal_threshold": 0.01},
        weight=1500.0,
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    action_rate = RewTerm(
        func=mdp.lebai_arm_action_rate_l2,
        weight=-1e-4,
    )

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["joint_[1-6]"]
            ),
        },
        weight=-1e-4,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.02, "asset_cfg": SceneEntityCfg("object")}
    )

    goal_success = DoneTerm(
        func=mdp.lebai_v6_goal_reached,
        params={"goal_threshold": 0.01},
        time_out=False,
    )    


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    #  原本这里的权重是10000步之后变为-1e-1

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}   
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0   #状态机需要10秒
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
