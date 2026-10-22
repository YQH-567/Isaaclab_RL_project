#!/usr/bin/env python3
"""LM3 PPO Reach v2: feedback-following, bounded real-motion diagnostic pilot.

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
MAX_EXCURSION_RAD = 0.30         # unverified joint envelope; path still needs human review
MAX_STEP_RAD = 0.005             # max change of commanded target per pilot tick
MAX_MEASURED_SPEED_RAD_S = 0.20  # stop on feedback exceeding this (not prevention)
MAX_FOLLOWING_ERROR_RAD = 0.12
MAX_TARGET_LEAD_RAD = 0.05      # commanded target can lead actual joint by <= 0.05rad
STARTUP_DEFAULT_TOL_RAD = 0.16 # require initial pose close to training default
JOINT_LIMIT_MARGIN_RAD = 0.10
COMMAND_V_RAD_S = 0.03
COMMAND_A_RAD_S2 = 0.08
MAX_READ_SECONDS = 0.07
MAX_SEND_SECONDS = 0.07

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
    low = JOINT_LIMITS[:, 0] + JOINT_LIMIT_MARGIN_RAD
    high = JOINT_LIMITS[:, 1] - JOINT_LIMIT_MARGIN_RAD
    if not np.all((q > low) & (q < high)):
        raise RuntimeError(f"Joint is at/near configured soft limit: {q}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
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
    if np.max(np.abs(start_q - required_default)) > STARTUP_DEFAULT_TOL_RAD:
        raise RuntimeError("Start pose too far from trained default; reposition through normal teaching mode")

    hz = EXEC_HZ if args.execute else args.hz
    period = 1.0 / hz
    print("[MODE]", "BOUNDED LIVE PILOT" if args.execute else "READ-ONLY DRY RUN")
    print("[INFO] Frequency:", hz, "Hz; duration:", args.duration, "s")
    print("[INFO] Initial joints:", start_q.tolist())
    print("[INFO] Target command:", DEFAULT_TARGET_POSE.tolist())
    print("[NOTE] The target is not yet geometrically validated against real flange/root frames.")
    if args.execute:
        print(f"[LIMITS] Max excursion per joint: ±{MAX_EXCURSION_RAD:.3f} rad from startup")
        print(f"[LIMITS] Max setpoint increment: {MAX_STEP_RAD:.3f} rad each {period * 1000:.0f} ms")
        print(f"[LIMITS] Maximum target lead from measured position: {MAX_TARGET_LEAD_RAD:.3f} rad")
        print(f"[LIMITS] Fixed SDK v={COMMAND_V_RAD_S:.3f} rad/s, a={COMMAND_A_RAD_S2:.3f} rad/s² (not guaranteed measured caps)")
        print("[WARNING] Robot controller limits and collision detection must be enabled;")
        print("          physical emergency stop and trained on-site supervision are required.")
        print("[IMPORTANT] Diagnostic flange error is NOT a validated reach criterion.")
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
    result = "time_window_completed"
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
            # Limit desired motion per tick AND prevent accumulating target lead.
            commanded = np.clip(bounded, previous_command - MAX_STEP_RAD, previous_command + MAX_STEP_RAD)
            if args.execute:
                commanded = np.clip(commanded, q - MAX_TARGET_LEAD_RAD, q + MAX_TARGET_LEAD_RAD)
            commanded = np.clip(commanded, start_q - MAX_EXCURSION_RAD, start_q + MAX_EXCURSION_RAD)
            commanded = np.clip(commanded, JOINT_LIMITS[:, 0] + JOINT_LIMIT_MARGIN_RAD,
                                JOINT_LIMITS[:, 1] - JOINT_LIMIT_MARGIN_RAD)
            if args.execute and np.max(np.abs(commanded - q)) > MAX_TARGET_LEAD_RAD + 1e-8:
                raise RuntimeError("Feedback target-lead invariant violated")
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
                   "max_command_step_rad": cmd_step_rad,
                   "max_command_lead_rad": float(np.max(np.abs(commanded - q))),
                   "max_tracking_error_rad": float(np.max(np.abs(previous_command - q))),
                   "at_excursion_boundary": int(np.any(np.abs(commanded - start_q) > MAX_EXCURSION_RAD - 0.001))}
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
                # Read-only post-stop sampling: no new motion commands.
                # Stop request is not proof of instantaneous stationary state.
                still_since = None
                stop_deadline = time.monotonic() + 2.0
                while time.monotonic() < stop_deadline:
                    post = robot.get_kin_data()
                    post_q = vector6(post["actual_joint_pose"], "post-stop joint pose")
                    post_dq = vector6(post["actual_joint_speed"], "post-stop speed")
                    if np.max(np.abs(post_dq)) < 0.015:
                        if still_since is None:
                            still_since = time.monotonic()
                        if time.monotonic() - still_since >= 0.30:
                            break
                    else:
                        still_since = None
                    time.sleep(0.05)
                post_error = np.linalg.norm(flange_position(post) - DEFAULT_TARGET_POSE[:3])
                print(f"[POST-STOP] q={post_q.round(4).tolist()} | max|dq|={np.max(np.abs(post_dq)):.4f} rad/s")
                print(f"[POST-STOP] diagnostic flange position error={post_error:.4f}m (uncalibrated)")
                if still_since is None or time.monotonic() - still_since < 0.30:
                    print("[WARNING] Stationary state not confirmed; inspect robot physically")
            except Exception as exc:
                print("[CRITICAL] Stop or post-stop verification failed; follow physical emergency procedure:", repr(exc))
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
        print("[NOTE] This pilot NEVER declares Reach success: flange/root/link_6 transform is unverified")
        print("[NOTE] Do not run unattended; operator must remain at physical E-stop")
    return 0 if result == "time_window_completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
