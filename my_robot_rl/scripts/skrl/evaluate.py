
"""Evaluate LM3 Reach PPO checkpoint (skrl / Isaac Lab 2.x)."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate LM3 Reach.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_trials", type=int, default=10)
parser.add_argument("--threshold", type=float, default=0.05)
parser.add_argument("--hold_time", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Sim 启动后才能导入依赖仿真的模块
import csv
import os
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.utils.math import combine_frame_transforms
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from skrl.utils.runner.torch import Runner

# 注册自己的 Gym 任务
import my_robot_rl.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, experiment_cfg):
    if args_cli.num_trials <= 0:
        raise ValueError("--num_trials must be positive")

    if args_cli.threshold <= 0 or args_cli.hold_time <= 0:
        raise ValueError("Threshold and hold_time must be positive")

    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    # 每次测试一个环境，一个目标
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 4.0
    env_cfg.commands.ee_pose.resampling_time_range = (4.0, 4.0)

    # 关闭观测噪声：先评估基础策略表现
    env_cfg.observations.policy.enable_corruption = False

    env_cfg.seed = args_cli.seed
    experiment_cfg["seed"] = args_cli.seed

    # 不生成新的训练日志和 checkpoint
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = raw_env.unwrapped

    # 定位 LM3 末端连杆
    robot = base_env.scene["robot"]
    body_ids, _ = robot.find_bodies("link_6")
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one link_6, got {body_ids}")
    ee_body_id = body_ids[0]

    # 与官方 Reach 奖励使用同一坐标系
    def get_position_error():
        command = base_env.command_manager.get_command("ee_pose")
        target_pos_b = command[:, :3]

        target_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            target_pos_b,
        )

        actual_pos_w = robot.data.body_pos_w[:, ee_body_id]

        return torch.linalg.vector_norm(
            target_pos_w - actual_pos_w,
            dim=-1,
        )[0].item()

    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
    runner = Runner(env, experiment_cfg)

    runner.agent.load(str(checkpoint))
    runner.agent.set_running_mode("eval")

    # 每次 env.step() 对应的仿真时间
    dt = base_env.step_dt
    steps_per_trial = int(round(4.0 / dt))
    hold_steps = max(1, int(round(args_cli.hold_time / dt)))

    results = []

    print("\n========== LM3 REACH EVALUATION ==========")
    print(f"Checkpoint: {checkpoint}")
    print(f"Trials: {args_cli.num_trials}")
    print(f"Success threshold: {args_cli.threshold:.3f} m")
    print(f"Hold time: {hold_steps * dt:.3f} s")
    print(f"Control dt: {dt:.6f} s")
    print("==========================================\n")

    try:
        for trial in range(args_cli.num_trials):
            # 每次显式 Reset，确保新起点、新目标
            obs, _ = env.reset()

            consecutive = 0
            reached_at = None
            success = False
            min_error = float("inf")
            final_error = float("inf")

            with torch.inference_mode():
                for step in range(steps_per_trial):
                    # 先记录当前目标误差，避免在
                    # Episode 结束自动 Reset 后误读新目标
                    error = get_position_error()

                    min_error = min(min_error, error)
                    final_error = error

                    if error < args_cli.threshold:
                        consecutive += 1
                    else:
                        consecutive = 0

                    if consecutive >= hold_steps and reached_at is None:
                        success = True
                        reached_at = (step + 1) * dt
                        # 不提前 break，继续观察到测试结束
                        # 确保最终误差仍有记录

                    outputs = runner.agent.act(
                        obs,
                        timestep=0,
                        timesteps=0,
                    )
                    actions = outputs[-1].get(
                        "mean_actions", outputs[0]
                    )

                    obs, _, terminated, truncated, _ = env.step(actions)

                    if bool(torch.as_tensor(
                        terminated | truncated
                    ).any()):
                        break

            record = {
                "trial": trial + 1,
                "success": int(success),
                "min_error_m": round(min_error, 6),
                "final_error_m": round(final_error, 6),
                "reach_time_s": (
                    round(reached_at, 4)
                    if reached_at is not None else ""
                ),
            }
            results.append(record)

            status = "PASS" if success else "FAIL"
            print(
                f"[{trial + 1:03d}] {status} | "
                f"min={min_error:.4f} m | "
                f"final={final_error:.4f} m | "
                f"time={reached_at}"
            )

    finally:
        env.close()

    # 汇总评估结果
    success_count = sum(r["success"] for r in results)
    success_rate = success_count / len(results)

    mean_min_error = sum(
        r["min_error_m"] for r in results
    ) / len(results)

    mean_final_error = sum(
        r["final_error_m"] for r in results
    ) / len(results)

    success_times = [
        r["reach_time_s"] for r in results
        if r["success"]
    ]

    mean_reach_time = (
        sum(success_times) / len(success_times)
        if success_times else None
    )

    print("\n========== EVALUATION RESULTS ==========")
    print(f"Success: {success_count}/{len(results)}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Mean min error: {mean_min_error:.4f} m")
    print(f"Mean final error: {mean_final_error:.4f} m")
    print(f"Mean reach time: {mean_reach_time}")
    print("========================================\n")

    # 保存独立评估记录
    output_dir = Path("evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "lm3_reach_evaluation.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial",
                "success",
                "min_error_m",
                "final_error_m",
                "reach_time_s",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
