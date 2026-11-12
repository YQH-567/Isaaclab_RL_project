from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


PROJECT_ROOT = Path(__file__).resolve().parents[4]

LM3_LMG90_USD_PATH = (
    PROJECT_ROOT
    #/ "assets/robots/lm3/usd/lebai_lm3_lmg90_gripper.usd"
    / "assets/robots/lm3/usd/lebai_lm3_lmg90_new1112.usd"
)


LEBAI_LM3_LMG90_CFG = ArticulationCfg(

    spawn=sim_utils.UsdFileCfg(
        usd_path=str(LM3_LMG90_USD_PATH),
        activate_contact_sensors=True,
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),

        joint_pos={
            # arm
            "joint_1": 3.13,
            "joint_2": -1.97,
            "joint_3": 1.71,
            "joint_4": -1.29,
            "joint_5": -1.62,
            "joint_6": 0.0,

            # gripper
            "gripper_r_joint1": 1.0,
            "gripper_r_joint2": -1.0,
            "gripper_r_joint_finger": -1.0,
            "gripper_l_joint1": 1.0,
            "gripper_l_joint2": -1.0,
            "gripper_l_joint_finger": -1.0,
        },
    ),

    actuators={

        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-6]"],
            stiffness=None,
            damping=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        ),

        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_r_joint1"],
            stiffness=None,
            damping=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        ),
    },
)