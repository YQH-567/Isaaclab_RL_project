
import sys
from pathlib import Path

import numpy as np
import lebai_sdk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.control.adapter import LM3Adapter


def main():
    # 使用实际的机械臂 IP
    robot_ip = "192.168.0.50"

    lebai_sdk.init()
    robot = lebai_sdk.connect(robot_ip, False)

    adapter = LM3Adapter()

    kin_data = robot.get_kin_data()

    obs12 = adapter.get_proprioception(kin_data)

    print("Joint relative position:", obs12[:6])
    print("Joint actual velocity:", obs12[6:12])
    print("Proprioception shape:", obs12.shape)

    assert obs12.shape == (12,)
    assert np.isfinite(obs12).all()

    print("[PASS] Real robot proprioception test")


if __name__ == "__main__":
    main()
