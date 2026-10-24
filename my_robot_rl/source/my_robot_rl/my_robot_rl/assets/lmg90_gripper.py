
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


# my_robot_rl 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[4]

LMG90_USD_PATH = (
    PROJECT_ROOT
    / "assets/grapper/lmg90/usd/lmg90_gripper_inertial_stable.usd"
)

LMG90_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(LMG90_USD_PATH),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            #"joint_[1-6]": 0.0,#这是一开始的，都设为0
            #下面是我重新设置的初始角度(0,-113,98,-74,-93,0)
            "gripper_r_joint1": 0.0,
            "gripper_r_joint2": 0.0,
            "gripper_r_joint_finger": 0.0,
            "gripper_l_joint1": 0.0,
            "gripper_l_joint2": 0.0,
            "gripper_l_joint_finger": 0.0,
        },
    ),

    actuators={
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_r_joint1"],
            stiffness=None,
            damping=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        ),  #stiffness=None、damping=None 是继承 USD 配置
    },
)
