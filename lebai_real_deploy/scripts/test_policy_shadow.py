
import argparse
import sys
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
    args = parser.parse_args()

    model_path = (
        ROOT / "checkpoints" / "lm3_reach_policy.pt"
    )

    policy = torch.jit.load(
        str(model_path),
        map_location="cpu",
    )
    policy.eval()

    adapter = LM3Adapter()

    # 固定的测试目标：基座坐标系下的位置和四元数
    # [x, y, z, qw, qx, qy, qz]
    # 此处先按 Isaac Lab UniformPoseCommand 的常用约定构造，
    # 后续仍需与训练环境的实际 command 输出核对。
    target_pose = np.array(
        [0.30, 0.00, 0.275, 0.0, 0.0, 1.0, 0.0],
        dtype=np.float32,
    )

    lebai_sdk.init()
    robot = lebai_sdk.connect(args.ip, False)

    # 唯一的机器人 SDK 状态读取
    kin_data = robot.get_kin_data()

    obs = adapter.build_observation(
        kin_data,
        target_pose,
    )

    obs_tensor = torch.from_numpy(
        obs
    ).unsqueeze(0)

    with torch.inference_mode():
        action_tensor = policy(obs_tensor)

    action = action_tensor.squeeze(0).cpu().numpy()

    # 仅用于分析，不发送到机械臂
    q_target = adapter.action_to_joint_target(action)

    print("\n========== Observation ==========")
    print("Joint pos rel:", obs[0:6])
    print("Joint vel rel:", obs[6:12])
    print("Target pose:", obs[12:19])
    print("Last action:", obs[19:25])
    print("Obs shape:", obs_tensor.shape)

    print("\n========== Policy ==========")
    print("Raw PPO action:", action)
    print("Calculated joint target:", q_target)
    print("Current actual joints:",
          kin_data["actual_joint_pose"])

    # 查看归一化后有哪些维度被裁剪到边界
    with torch.inference_mode():
        normalized = (
            (obs_tensor - policy.mean)
            / (torch.sqrt(policy.variance) + policy.epsilon)
        )

    clipped_indices = torch.nonzero(
        torch.abs(normalized[0]) > policy.clip
    ).flatten().tolist()

    print("Normalization clipped indices:", clipped_indices)
    print("\n[PASS] Shadow inference completed")
    print("[SAFE] No motion command was sent")


if __name__ == "__main__":
    main()
