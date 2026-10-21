
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import lebai_sdk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.control.adapter import LM3Adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--hz", type=float, default=30.0)
    args = parser.parse_args()

    if args.duration <= 0 or args.hz <= 0:
        raise ValueError("duration and hz must be positive")

    # 1. 加载模型
    model_path = ROOT / "checkpoints/lm3_reach_policy.pt"

    policy = torch.jit.load(
        str(model_path),
        map_location="cpu",
    )
    policy.eval()

    # 2. 初始化 Adapter
    adapter = LM3Adapter()

    # 固定 Reach 目标
    # [x, y, z, qw, qx, qy, qz]
    target_pose = np.array(
        [0.30, 0.0, 0.275, 0.0, 0.0, 1.0, 0.0],
        dtype=np.float32,
    )

    # 3. SDK 连接，只读取状态
    lebai_sdk.init()
    robot = lebai_sdk.connect(args.ip, False)

    period = 1.0 / args.hz
    records = []

    start = time.perf_counter()
    next_tick = start
    last_tick = None

    print("[INFO] Starting read-only shadow mode")
    print(f"[INFO] Target frequency: {args.hz} Hz")

    try:
        while time.perf_counter() - start < args.duration:
            now = time.perf_counter()

            if now < next_tick:
                time.sleep(next_tick - now)

            tick_start = time.perf_counter()

            # 4. 读取真实状态
            read_start = time.perf_counter()
            kin_data = robot.get_kin_data()
            read_ms = (
                time.perf_counter() - read_start
            ) * 1000.0

            # 5. 构造 Observation
            obs = adapter.build_observation(
                kin_data,
                target_pose,
            )

            obs_tensor = torch.from_numpy(
                obs
            ).unsqueeze(0)

            # 6. PPO 推理
            inference_start = time.perf_counter()

            with torch.inference_mode():
                action = policy(obs_tensor)

            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000.0

            action_np = action.squeeze(0).numpy()

            # 7. 计算目标角（仅记录，绝不发送）
            q_target = adapter.action_to_joint_target(
                action_np
            )

            # 8. 更新上一次 Action
            adapter.update_last_action(action_np)

            # 9. 统计周期
            loop_ms = (
                time.perf_counter() - tick_start
            ) * 1000.0

            dt_ms = (
                (tick_start - last_tick) * 1000.0
                if last_tick is not None
                else 0.0
            )
            last_tick = tick_start

            record = {
                "elapsed_s": tick_start - start,
                "dt_ms": dt_ms,
                "read_ms": read_ms,
                "inference_ms": inference_ms,
                "loop_ms": loop_ms,
            }

            for i in range(6):
                record[f"q{i+1}"] = float(
                    kin_data["actual_joint_pose"][i]
                )
                record[f"qd{i+1}"] = float(
                    kin_data["actual_joint_speed"][i]
                )
                record[f"action{i+1}"] = float(
                    action_np[i]
                )
                record[f"target{i+1}"] = float(
                    q_target[i]
                )

            records.append(record)

            next_tick += period

            # 如果明显超时，跳过过期的调度点
            if time.perf_counter() > next_tick:
                next_tick = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")

    finally:
        log_dir = ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / "shadow_loop.csv"

        if records:
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=records[0].keys(),
                )
                writer.writeheader()
                writer.writerows(records)

        print("\n========== Results ==========")
        print("Samples:", len(records))
        print("Log:", log_path)

        if len(records) > 1:
            intervals = np.array([
                r["dt_ms"] for r in records[1:]
            ])

            reads = np.array([
                r["read_ms"] for r in records
            ])

            inferences = np.array([
                r["inference_ms"] for r in records
            ])

            print("Mean period (ms):", intervals.mean())
            print("Max period (ms):", intervals.max())
            print("Mean SDK read (ms):", reads.mean())
            print("Max SDK read (ms):", reads.max())
            print("Mean inference (ms):", inferences.mean())

        print("[PASS] Shadow loop finished")
        print("[SAFE] No motion commands were sent")


if __name__ == "__main__":
    main()
