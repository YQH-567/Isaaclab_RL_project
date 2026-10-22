
# import numpy as np


# class LM3Adapter:
#     def __init__(self):
#         # Isaac Lab 训练时的默认关节位置
#         self.q_default = np.array(
#             [0.0, -1.97, 1.71, -1.29, -1.62, 0.0],
#             dtype=np.float32,
#         )

#         # JointPositionActionCfg 中的 scale
#         self.action_scale = 0.5

#     def joint_position_observation(self, q_real):
#         """实机关节角 -> 相对默认关节位置。"""
#         q_real = np.asarray(q_real, dtype=np.float32)

#         if q_real.shape != (6,):
#             raise ValueError(
#                 f"Expected (6,), got {q_real.shape}"
#             )

#         return q_real - self.q_default

#     def action_to_joint_target(self, action):
#         """PPO Action -> 仿真定义的绝对关节目标。

#         注意：只进行数学转换，不发送机器人命令。
#         """
#         action = np.asarray(action, dtype=np.float32)

#         if action.shape != (6,):
#             raise ValueError(
#                 f"Expected (6,), got {action.shape}"
#             )

#         if not np.isfinite(action).all():
#             raise ValueError("Invalid action values")

#         return self.q_default + self.action_scale * action



#!/usr/bin/env python3
"""LM3 单关节小幅运动测试。

默认只读检查。只有指定 --execute 并通过交互确认才会发送运动指令。
本脚本不是完整的工业安全控制器。
"""

import argparse
import math
import sys
import time

import lebai_sdk


# 来自你提供的 LM3 网页安全设置截图，单位 rad
JOINT_LIMITS = [
    (-6.28, 6.28),   # J1
    (-3.49, 0.35),   # J2
    (-3.14, 3.14),   # J3
    (-6.28, 6.28),   # J4
    (-6.28, 6.28),   # J5
    (-6.28, 6.28),   # J6
]

LIMIT_MARGIN = 0.05       # 距配置限位至少留 0.05 rad
MAX_DELTA = 0.03          # 单次测试角度变化不超过约 1.72°
MAX_VELOCITY = 0.05       # rad/s
MAX_ACCELERATION = 0.10   # rad/s²


def six_floats(values, name):
    """检查 SDK 返回的六轴数据。"""
    if values is None or len(values) != 6:
        raise ValueError(f"{name} 必须包含 6 个关节值")

    result = [float(x) for x in values]
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} 包含 NaN 或 Inf")

    return result


def safe_stop(robot):
    """尽力停止运动；停止失败时明确报告。"""
    try:
        robot.stop()
        print("[STOP] 已发送 robot.stop()")
    except Exception as exc:
        print(f"[CRITICAL] 停止指令调用失败：{exc}")
        print("请立即按现场安全规程处理，必要时使用实体急停。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.0.50")
    parser.add_argument("--joint", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--delta", type=float, default=0.02,
                        help="目标关节角变化量，单位 rad，可正可负")
    parser.add_argument("--velocity", type=float, default=0.05,
                        help="关节速度，单位 rad/s")
    parser.add_argument("--acceleration", type=float, default=0.10,
                        help="关节加速度，单位 rad/s^2")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="运动等待超时，单位秒")
    parser.add_argument("--execute", action="store_true",
                        help="明确开启运动测试；默认只读")
    args = parser.parse_args()

    # 限制首次测试参数，避免误输入过大的数值。
    if not 0 < abs(args.delta) <= MAX_DELTA:
        parser.error(f"|delta| 必须在 (0, {MAX_DELTA}] rad 内")

    if not 0 < args.velocity <= MAX_VELOCITY:
        parser.error(f"velocity 必须在 (0, {MAX_VELOCITY}] rad/s 内")

    if not 0 < args.acceleration <= MAX_ACCELERATION:
        parser.error(
            f"acceleration 必须在 (0, {MAX_ACCELERATION}] rad/s² 内"
        )

    if not 0.5 <= args.timeout <= 10:
        parser.error("timeout 必须在 0.5 到 10 秒之间")

    lebai_sdk.init()
    robot = lebai_sdk.connect(args.ip, False)

    command_sent = False

    try:
        data = robot.get_kin_data()

        current = six_floats(
            data["actual_joint_pose"], "actual_joint_pose"
        )
        velocity = six_floats(
            data["actual_joint_speed"], "actual_joint_speed"
        )

        print("\n========== LM3 Joint Test ==========")
        print("Current joints (rad):", current)
        print("Joint velocities (rad/s):", velocity)
        print("Joint limits:", JOINT_LIMITS)

        # 当前状态必须处于允许范围内，并与边界留出余量。
        for i, (q, (lo, hi)) in enumerate(
            zip(current, JOINT_LIMITS), start=1
        ):
            if not lo + LIMIT_MARGIN < q < hi - LIMIT_MARGIN:
                raise RuntimeError(
                    f"J{i} 当前角度 {q:.4f} rad "
                    "接近或超出配置限位，拒绝运动"
                )

        # 确保机械臂处于静止状态。
        if max(abs(v) for v in velocity) > 0.02:
            raise RuntimeError(
                "检测到关节仍在运动，等待机械臂静止后再测试"
            )

        index = args.joint - 1
        target = current.copy()
        target[index] += args.delta

        lo, hi = JOINT_LIMITS[index]
        if not lo + LIMIT_MARGIN < target[index] < hi - LIMIT_MARGIN:
            raise RuntimeError("目标关节角接近或超出配置限位")

        print(f"Selected joint: J{args.joint}")
        print(f"Start angle: {current[index]:.6f} rad")
        print(f"Target angle: {target[index]:.6f} rad")
        print(f"Delta: {args.delta:+.6f} rad")
        print(f"Velocity: {args.velocity:.3f} rad/s")
        print(f"Acceleration: {args.acceleration:.3f} rad/s²")

        if not args.execute:
            print("\n[DRY RUN] 未发送任何运动指令。")
            print("检查输出无误后，再按现场安全规程准备运动测试。")
            return 0

        print("\n运动前请确认：")
        print("- 机械臂周围无人，运动路径无障碍")
        print("- 末端负载和夹爪处于安全状态")
        print("- 实体急停可用，控制器限位已启用")
        print("- 其他五个关节应保持静止")

        confirmation = input(
            f"输入 MOVE J{args.joint} {args.delta:+.3f} "
            "确认发送指令："
        ).strip()

        expected = f"MOVE J{args.joint} {args.delta:+.3f}"
        if confirmation != expected:
            print("[ABORT] 确认文本不匹配，未发送运动指令。")
            return 0

        # 再次读取状态，避免确认期间机器人位置已改变。
        latest = robot.get_kin_data()
        latest_pose = six_floats(
            latest["actual_joint_pose"], "actual_joint_pose"
        )
        latest_speed = six_floats(
            latest["actual_joint_speed"], "actual_joint_speed"
        )

        if max(abs(latest_pose[i] - current[i]) for i in range(6)) > 0.01:
            raise RuntimeError("确认期间关节位置发生变化，拒绝运动")

        if max(abs(v) for v in latest_speed) > 0.02:
            raise RuntimeError("确认期间检测到关节运动，拒绝运动")

        # 使用 SDK 提供的接口启用关节限位。
        robot.enable_joint_limits()

        # 发送一次小幅单关节目标。
        command_sent = True
        motion_id = robot.towardj(
            target, args.acceleration, args.velocity
        )
        print(f"[INFO] Motion command sent. ID: {motion_id}")

        start_time = time.monotonic()
        last_report = 0.0
        stable_count = 0
        max_abs_joint_speed = 0.0

        while True:
            elapsed = time.monotonic() - start_time

            if elapsed > args.timeout:
                raise TimeoutError("等待目标到达超时")

            data = robot.get_kin_data()
            actual = six_floats(
                data["actual_joint_pose"], "actual_joint_pose"
            )
            speeds = six_floats(
                data["actual_joint_speed"], "actual_joint_speed"
            )

            max_abs_joint_speed = max(
            max_abs_joint_speed,
            abs(speeds[index]),
            )

            error = abs(target[index] - actual[index])
            other_deviation = max(
                abs(actual[i] - current[i])
                for i in range(6) if i != index
            )

            if other_deviation > 0.03:
                raise RuntimeError(
                    "其他关节偏移超过 0.03 rad，停止测试"
                )

            if error <= 0.005 and abs(speeds[index]) <= 0.02:
                stable_count += 1
            else:
                stable_count = 0

            if elapsed - last_report >= 0.25:
                print(
                    f"t={elapsed:.2f}s | "
                    f"J{args.joint}={actual[index]:.5f} rad | "
                    f"error={error:.5f} rad | "
                    f"velocity={speeds[index]:.4f} rad/s"
                )
                last_report = elapsed

            if stable_count >= 3:
                print("\n[PASS] 目标位置已达到，关节速度已降低。")
                print("Final joints (rad):", actual)


                print(
                    f"Maximum observed J{args.joint} speed: "
                    f"{max_abs_joint_speed:.4f} rad/s"
                )
                print(
                    f"Commanded velocity parameter: "
                    f"{args.velocity:.4f} rad/s"
                )
                return 0

            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] 收到键盘中断。")
        if command_sent:
            safe_stop(robot)
        return 130

    except Exception as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        if command_sent:
            safe_stop(robot)
        return 1


if __name__ == "__main__":
    sys.exit(main())
