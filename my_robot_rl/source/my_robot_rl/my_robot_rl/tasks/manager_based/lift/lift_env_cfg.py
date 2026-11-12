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

    # V6: one-time reward when reaching the goal
    v6_goal_success = RewTerm(
        func=mdp.lebai_v7_success_bonus,
        params={"goal_threshold": 0.01,"speed_std": 0.1},
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
        #weight=-1e-0,
        weight=-10.0
    )

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["joint_[1-6]"]
            ),
        },
        # weight=-1e-0,
        weight=-10.0
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.02, "asset_cfg": SceneEntityCfg("object")}
    )

    goal_success = DoneTerm(
        func=mdp.lebai_v7_goal_reached,
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
