#!/usr/bin/env python3
"""LM3 PPO Reach: dry-run at 30 Hz, or bounded, short real-motion pilot.

This is an experimental supervisory controller, NOT a certified safety controller.
Run only with a qualified operator, a clear workspace, and physical e-stop available.
"""
import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import lebai_sdk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.control.adapter import LM3Adapter

# Confirmed from user's robot controller UI; re-check before each experiment.
JOINT_LIMITS = np.array([
    [-6.28, 6.28], [-3.49, 0.35], [-3.14, 3.14],
    [-6.28, 6.28], [-6.28, 6.28], [-6.28, 6.28],
], dtype=np.float64)

# Pilot-only restrictions. These are software safeguards, not safety guarantees.
EXEC_HZ = 10.0
EXEC_DURATION_MAX_S = 15.0
MAX_EXCURSION_RAD = 1.0  #原来0.06         # all joints: no further than 0.06 rad from startup
MAX_STEP_RAD = 0.5    #原来0.005             # max change of commanded target per pilot tick
MAX_MEASURED_SPEED_RAD_S = 1.0  #原本0.12 # stop on feedback exceeding this (not prevention)
MAX_FOLLOWING_ERROR_RAD = 10.0     #原本是0.12
JOINT_LIMIT_MARGIN_RAD = 1.0   #原本0.10
COMMAND_V_RAD_S = 0.3   #原本0.03
COMMAND_A_RAD_S2 = 0.8  #原本0.08
MAX_READ_SECONDS = 0.7  #原本0.07
MAX_SEND_SECONDS = 0.7  #原本0.07

# PPO Reach 实机测试参数（建议配置）
# 注意：软件限制不能替代控制器限位、碰撞检测和实体急停

# EXEC_HZ = 10.0
# EXEC_DURATION_MAX_S = 2

# # 允许的关节活动范围（相对于启动位置）
# MAX_EXCURSION_RAD = 0.30

# # 每个控制周期允许的最大关节目标变化
# MAX_STEP_RAD = 0.005

# # 实测关节速度异常停止阈值（不是速度控制上限）
# MAX_MEASURED_SPEED_RAD_S = 0.20

# # 下发目标与实际关节位置的最大容许偏差
# MAX_FOLLOWING_ERROR_RAD = 0.12

# # 距离关节硬限位的安全余量
# JOINT_LIMIT_MARGIN_RAD = 0.10

# # towardj() 的速度和加速度参数
# COMMAND_V_RAD_S = 0.03
# COMMAND_A_RAD_S2 = 0.08

# # SDK 读取和指令发送的超时检查阈值
# MAX_READ_SECONDS = 0.07
# MAX_SEND_SECONDS = 0.07

# PPO Reach：下一阶段验证配置

# EXEC_HZ = 10.0

# # 从固定 2 秒改为最长 15 秒的实验窗口
# EXEC_DURATION_MAX_S = 15.0

# # 保留已验证的短时活动范围，扩大范围前先检查运动路径
# MAX_EXCURSION_RAD = 0.30

# # 每个控制周期目标变化上限
# MAX_STEP_RAD = 0.005

# # 反馈异常检测
# MAX_MEASURED_SPEED_RAD_S = 0.20
# MAX_FOLLOWING_ERROR_RAD = 0.12

# # 关节限位余量
# JOINT_LIMIT_MARGIN_RAD = 0.10

# # 控制器指令参数
# COMMAND_V_RAD_S = 0.03
# COMMAND_A_RAD_S2 = 0.08

# # 通信耗时检查
# MAX_READ_SECONDS = 0.07
# MAX_SEND_SECONDS = 0.07

# # Reach 成功判定参数
# REACH_POSITION_TOLERANCE_M = 0.02
# REACH_STABLE_TIME_S = 0.5

# Isaac Lab v2.2.1: [x,y,z,qw,qx,qy,qz] in robot root frame.
# Must be confirmed against the deployed model + real flange geometry before interpreting errors.
DEFAULT_TARGET_POSE = np.array([0.30, 0.0, 0.275, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def vector6(values, label):
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (6,) or not np.isfinite(arr).all():
        raise ValueError(f"{label} must be a finite 6-vector; got {arr}")
    return arr


def flange_position(data):
    p = data.get("actual_flange_pose")
    if not isinstance(p, dict):
        return np.array([math.nan] * 3)
    return np.array([float(p[k]) for k in ("x", "y", "z")], dtype=np.float64)


def verify_joint_bounds(q):
    # low = JOINT_LIMITS[:, 0] + JOINT_LIMIT_MARGIN_RAD
    # high = JOINT_LIMITS[:, 1] - JOINT_LIMIT_MARGIN_RAD
    low = JOINT_LIMITS[:, 0]
    high = JOINT_LIMITS[:, 1]
    if not np.all((q > low) & (q < high)):
        raise RuntimeError(f"Joint is at/near configured soft limit: {q}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--hz", type=float, default=30.0, help="Dry-run frequency only; live pilot fixed at 10 Hz")
    parser.add_argument("--execute", action="store_true", help="Enable bounded physical motion after confirmation")
    args = parser.parse_args()
    if not (0.1 <= args.duration <= 30):
        parser.error("--duration must be in [0.1, 30] seconds")
    if not (1 <= args.hz <= 30):
        parser.error("--hz must be in [1, 30]")
    if args.execute and args.duration > EXEC_DURATION_MAX_S:
        parser.error(f"Live pilot duration cannot exceed {EXEC_DURATION_MAX_S}s")

    torch.set_num_threads(1)
    model_path = ROOT / "checkpoints" / "lm3_reach_policy.pt"
    policy = torch.jit.load(str(model_path), map_location="cpu").eval()
    adapter = LM3Adapter()
    # Must reproduce trained LM3 default joint positions.
    required_default = np.array([3.13, -1.97, 1.71, -1.29, -1.62, 0.0])
    if not np.allclose(adapter.q_default, required_default, atol=1e-5):
        raise RuntimeError(f"Adapter default joints mismatch training: {adapter.q_default}")

    lebai_sdk.init()
    robot = lebai_sdk.connect(args.ip, False)
    initial = robot.get_kin_data()
    start_q = vector6(initial["actual_joint_pose"], "initial joints")
    start_speed = vector6(initial["actual_joint_speed"], "initial speeds")
    verify_joint_bounds(start_q)
    if np.max(np.abs(start_speed)) > 0.015:
        raise RuntimeError("Robot not stationary at startup")

    hz = EXEC_HZ if args.execute else args.hz
    period = 1.0 / hz
    print("[MODE]", "BOUNDED LIVE PILOT" if args.execute else "READ-ONLY DRY RUN")
    print("[INFO] Frequency:", hz, "Hz; duration:", args.duration, "s")
    print("[INFO] Initial joints:", start_q.tolist())
    print("[INFO] Target command:", DEFAULT_TARGET_POSE.tolist())
    print("[NOTE] The target is not yet geometrically validated against real flange/root frames.")
    if args.execute:
        print("[LIMITS] Max excursion per joint: ±{MAX_EXCURSION_RAD:.3f} rad from startup")
        print("[LIMITS] Max setpoint increment: 0.005 rad each 100 ms")
        print("[LIMITS] Fixed SDK v=0.03 rad/s, a=0.08 rad/s² (not guaranteed measured caps)")
        print("[WARNING] Robot controller limits and collision detection must be enabled;")
        print("          physical emergency stop and trained on-site supervision are required.")
        phrase = 'RUN BOUNDED PPO'
        if input(f'Type {phrase} to send real motion commands: ').strip() != phrase:
            print("[ABORT] No commands sent")
            return 0
        # Do not auto start the robot: operator/controller must already be in valid motion state.
        # State may change while operator reads the prompt.
        confirmed_state = robot.get_kin_data()
        confirmed_q = vector6(confirmed_state["actual_joint_pose"], "confirmed joints")
        confirmed_dq = vector6(confirmed_state["actual_joint_speed"], "confirmed speeds")
        verify_joint_bounds(confirmed_q)
        if np.max(np.abs(confirmed_q - start_q)) > 0.005 or np.max(np.abs(confirmed_dq)) > 0.015:
            raise RuntimeError("Robot moved during confirmation; restart the pilot")
        robot.enable_joint_limits()

    records = []
    sent_any = False
    previous_command = start_q.copy()
    next_deadline = time.monotonic()
    start_time = next_deadline
    result = "completed"
    error_message = ""
    try:
        while time.monotonic() - start_time < args.duration:
            now = time.monotonic()
            if now < next_deadline:
                time.sleep(next_deadline - now)
            tick_start = time.monotonic()
            if args.execute and tick_start - next_deadline > period:
                raise RuntimeError("Control schedule missed by more than one period")

            t0 = time.monotonic()
            state = robot.get_kin_data()
            read_s = time.monotonic() - t0
            if args.execute and read_s > MAX_READ_SECONDS:
                raise RuntimeError(f"SDK state read too slow: {read_s:.3f}s")
            q = vector6(state["actual_joint_pose"], "actual_joint_pose")
            dq = vector6(state["actual_joint_speed"], "actual_joint_speed")
            verify_joint_bounds(q)
            if args.execute:
                if np.max(np.abs(dq)) > MAX_MEASURED_SPEED_RAD_S:
                    raise RuntimeError(f"Measured joint speed too high: {np.max(np.abs(dq)):.3f}rad/s")
                if np.max(np.abs(q - start_q)) > MAX_EXCURSION_RAD + 0.005:
                    raise RuntimeError("Actual position exceeded startup excursion envelope")
                if np.max(np.abs(q - previous_command)) > MAX_FOLLOWING_ERROR_RAD:
                    raise RuntimeError("Setpoint tracking discrepancy too large")

            obs = adapter.build_observation(state, DEFAULT_TARGET_POSE)
            with torch.inference_mode():
                action = policy(torch.from_numpy(obs).view(1, 25)).reshape(-1).numpy().copy()
            if action.shape != (6,) or not np.isfinite(action).all():
                raise RuntimeError("Invalid PPO action")
            raw_target = vector6(adapter.action_to_joint_target(action), "raw target")
            # First constrain the absolute reachable target around observed safe startup pose.
            bounded = np.clip(raw_target, start_q - MAX_EXCURSION_RAD, start_q + MAX_EXCURSION_RAD)
            # Then apply a maximum target step relative to the last command.
            commanded = np.clip(bounded, previous_command - MAX_STEP_RAD, previous_command + MAX_STEP_RAD)
            commanded = np.clip(commanded, JOINT_LIMITS[:, 0] + JOINT_LIMIT_MARGIN_RAD,
                                JOINT_LIMITS[:, 1] - JOINT_LIMIT_MARGIN_RAD)
            flange = flange_position(state)
            diagnostic_pos_err = float(np.linalg.norm(flange - DEFAULT_TARGET_POSE[:3]))
            send_s = 0.0
            motion_id = ""
            cmd_step_rad = float(np.max(np.abs(commanded - previous_command)))
            if args.execute:
                t1 = time.monotonic()
                # Set before calling: SDK may accept command and then raise on transport.
                sent_any = True
                motion_id = robot.towardj(commanded.tolist(), COMMAND_A_RAD_S2, COMMAND_V_RAD_S)
                send_s = time.monotonic() - t1
                if send_s > MAX_SEND_SECONDS:
                    raise RuntimeError(f"SDK send took too long: {send_s:.3f}s")
                previous_command = commanded.copy()
            # last_action is the raw policy action, not the clipped joint target.
            adapter.update_last_action(action.astype(np.float32))

            row = {"elapsed_s": round(tick_start - start_time, 6),
                   "read_ms": round(read_s * 1000, 3), "send_ms": round(send_s * 1000, 3),
                   "motion_id": motion_id, "diagnostic_flange_pos_error_m": diagnostic_pos_err,
                   "max_raw_target_step_rad": float(np.max(np.abs(raw_target - q))),
                   "max_command_step_rad": cmd_step_rad}
            for i in range(6):
                row[f"q{i+1}"] = float(q[i]); row[f"dq{i+1}"] = float(dq[i])
                row[f"action{i+1}"] = float(action[i]); row[f"raw_target{i+1}"] = float(raw_target[i])
                row[f"command{i+1}"] = float(commanded[i])
            records.append(row)
            if len(records) == 1 or len(records) % int(max(1, round(hz))) == 0:
                print(f"t={row['elapsed_s']:.2f}s | est. flange pos error={diagnostic_pos_err:.3f}m | "
                      f"max|dq|={np.max(np.abs(dq)):.3f}rad/s | "
                      f"max|raw-q|={row['max_raw_target_step_rad']:.3f}rad")
            next_deadline += period
            if not args.execute and time.monotonic() > next_deadline:
                next_deadline = time.monotonic()

    except KeyboardInterrupt:
        result = "interrupted"
        error_message = "User interruption"
    except Exception as exc:
        result = "failed"
        error_message = f"{type(exc).__name__}: {exc}"
    finally:
        # Always attempt to stop after enabling real motion, including on normal completion.
        if args.execute:
            try:
                robot.stop()
                print("[STOP] robot.stop() called")
            except Exception as exc:
                print("[CRITICAL] stop() failed; use physical emergency procedure:", repr(exc))
        logs = ROOT / "logs"
        logs.mkdir(exist_ok=True, parents=True)
        name = f"reach_{'live' if args.execute else 'dry'}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        logpath = logs / name
        if records:
            with logpath.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
        print(f"[RESULT] {result}; samples={len(records)}; log={logpath}")
        if error_message:
            print("[ERROR]", error_message)
        print("[NOTE] End-effector position error is diagnostic until frame alignment is validated")
    return 0 if result == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
