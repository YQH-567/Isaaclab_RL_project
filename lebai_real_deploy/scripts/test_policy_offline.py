
from pathlib import Path

import torch


def main():
    # 项目根目录
    project_root = Path(__file__).resolve().parents[1]

    model_path = (
        project_root
        / "checkpoints"
        / "lm3_reach_policy.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    # 1. 加载 TorchScript
    policy = torch.jit.load(
        str(model_path),
        map_location="cpu",
    )
    policy.eval()

    print("[INFO] Model loaded:", model_path)

    # 2. 构造测试 Observation
    # 注意：全零 Observation 只是离线加载测试
    obs = torch.zeros(
        (1, 25),
        dtype=torch.float32,
    )

    # 3. 确定性推理
    with torch.inference_mode():
        action = policy(obs)

    # 4. 检查输出
    assert action.shape == (1, 6), (
        f"Unexpected action shape: {action.shape}"
    )

    assert torch.isfinite(action).all(), (
        "Policy output contains NaN or Inf"
    )

    # 5. 重复推理，验证结果一致
    with torch.inference_mode():
        action_repeat = policy(obs)

    max_error = torch.max(
        torch.abs(action - action_repeat)
    ).item()

    print("Observation shape:", obs.shape)
    print("Action shape:", action.shape)
    print("Action:", action)
    print("Repeat inference error:", max_error)

    assert max_error < 1e-6

    print("[PASS] Offline policy inference successful")


if __name__ == "__main__":
    main()
