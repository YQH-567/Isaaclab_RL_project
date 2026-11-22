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


from . import mdp
#增加相机或者传感器
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils


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


    # Industrial parts tray / KLT bin
    tray = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tray",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -0.3, 0.03),     #我确认这里的位置是中心位置
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
            scale=(0.8, 0.8, 0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
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

    # # camera
    # # Wrist camera: Orbbec Astra Pro Plus RGB
    # wrist_camera = CameraCfg(
    #     # 必须替换为 LM3 USD 中真实存在的末端刚体 Link 路径
    #     prim_path="{ENV_REGEX_NS}/Robot/lebai_lm3/link_6/WristCamera",

    #     update_period=0.04,  # 10 Hz，10HZ训练出来发现不太行，现在换成25HZ
    #     height=180,
    #     width=320,
    #     data_types=["rgb", "distance_to_image_plane"],

    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=15.3,
    #         horizontal_aperture=20.955,
    #         clipping_range=(0.1, 10.0),
    #     ),

    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.0, 0.08, -0.02),
    #         #rot=(1.0, 0.0, 0.0, 0.0),
    #         # 相对 link_6 的 X 轴旋转 +10°
    #         #rot=(0.9962, 0.0872, 0.0, 0.0),
    #         rot=(0.9848, 0.1736, 0.0, 0.0),  #X 轴旋转 +20°
    #         convention="ros",
    #     ),
    # )

    # # Overhead camera: Intel RealSense D435 RGB
    # front_camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/FrontCamera",

    #     update_period=0.04,  #这里也是
    #     height=180,
    #     width=320,
    #     data_types=["rgb", "distance_to_image_plane"],

    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=15.3,
    #         horizontal_aperture=20.955,
    #         clipping_range=(0.1, 10.0),
    #     ),

    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.25, -0.00, 1.20),
    #         rot=(0.0, 1.0, 0.0, 0.0),
    #         convention="ros",
    #     ),
    # )

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
            pos_x=(0.0, 0.0), pos_y=(-0.3, -0.3), pos_z=(0.2, 0.2), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)    #这里是抬起的目标点，目前我固定一个点
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
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )



# reward V5版本和V6版本，这个V5其实跟没写是一样的，效果出来一模一样
@configclass
class RewardsCfg:
    """这个act应该都不涉及这个奖励函数"""


    action_rate = RewTerm(
        func=mdp.action_rate_l2,
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
        self.episode_length_s = 10.0   #place那个状态机需要10秒
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
