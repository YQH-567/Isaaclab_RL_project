
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


# my_robot_rl 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[4]

LM3_USD_PATH = (
    PROJECT_ROOT
    / "assets/robots/lm3/usd/lebai_lm3_PD_1.usd"
)


LEBAI_LM3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(LM3_USD_PATH),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint_[1-6]": 0.0,
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
    },
)
