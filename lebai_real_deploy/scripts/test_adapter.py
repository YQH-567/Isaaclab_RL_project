
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.control.adapter import LM3Adapter


adapter = LM3Adapter()

q_real = np.array([
    0.0,
    -1.969918953,
    1.709909209,
    -1.289981969,
    -1.622472305,
    0.0,
])

q_rel = adapter.joint_position_observation(q_real)

print("Joint relative positions:")
print(q_rel)

assert np.max(np.abs(q_rel)) < 0.01

# 只验证数学转换，不使用策略动作控制实机
test_action = np.zeros(6)

q_target = adapter.action_to_joint_target(test_action)

print("Zero-action joint target:")
print(q_target)

assert np.allclose(q_target, adapter.q_default)

print("[PASS] Adapter basic tests")
