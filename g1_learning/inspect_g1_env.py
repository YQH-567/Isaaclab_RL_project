"""检查 Isaac Lab G1 强化学习环境的结构。"""

import argparse

from isaaclab.app import AppLauncher


# ------------------------------------------------------------
# 1. 解析命令行参数
# ------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Inspect the Unitree G1 locomotion environment."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-G1-v0",
    help="Gymnasium task ID.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel environments.",
)

# 添加 --headless、--device、--disable_fabric 等 Isaac Lab 参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# ------------------------------------------------------------
# 2. 必须先启动 Isaac Sim，再导入其他 Isaac Lab 模块
# ------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ------------------------------------------------------------
# 3. Isaac Sim 启动后才能导入这些包
# ------------------------------------------------------------
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    # 根据任务名称读取 EnvCfg
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    print_section("环境配置")

    print("任务名称：", args_cli.task)
    print("并行环境数：", env_cfg.scene.num_envs)
    print("仿真设备：", env_cfg.sim.device)
    print("物理步长 sim.dt：", env_cfg.sim.dt)
    print("控制降采样 decimation：", env_cfg.decimation)
    print(
        "策略控制周期：",
        env_cfg.sim.dt * env_cfg.decimation,
        "秒",
    )
    print(
        "策略控制频率：",
        1.0 / (env_cfg.sim.dt * env_cfg.decimation),
        "Hz",
    )

    # gym.make 会根据注册信息创建 ManagerBasedRLEnv
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped

    # 获取场景中的机器人
    robot = base_env.scene["robot"]

    print_section("Gym 空间")

    print("Observation space：")
    print(env.observation_space)

    print()
    print("Action space：")
    print(env.action_space)

    print_section("机器人结构")

    print("关节数量：", len(robot.joint_names))
    for index, name in enumerate(robot.joint_names):
        print(f"joint[{index:02d}] = {name}")

    print()
    print("刚体数量：", len(robot.body_names))
    for index, name in enumerate(robot.body_names):
        print(f"body[{index:02d}] = {name}")

    print_section("ManagerBasedRLEnv 管理器")

    print("动作管理器：")
    print(base_env.action_manager)

    print()
    print("观测管理器：")
    print(base_env.observation_manager)

    print()
    print("指令管理器：")
    print(base_env.command_manager)

    print()
    print("奖励管理器：")
    print(base_env.reward_manager)

    print()
    print("终止管理器：")
    print(base_env.termination_manager)

    print()
    print("事件管理器：")
    print(base_env.event_manager)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
