
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


# 找到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[4]

USD_PATH = (
    PROJECT_ROOT
    / "assets/robots/lm3/usd/lebai_lm3.usd"
)


LEBAI_LM3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
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

            # 第一版先继承 USD 中的 Drive 增益
            stiffness=None,
            damping=None,
        ),
    },
)
