
import numpy as np


class LM3Adapter:
    def __init__(self):
        # Isaac Lab 训练时的默认关节位置
        self.q_default = np.array(
            [0.0, -1.97, 1.71, -1.29, -1.62, 0.0],
            dtype=np.float32,
        )

        # JointPositionActionCfg 中的 scale
        self.action_scale = 0.5

    def joint_position_observation(self, q_real):
        """实机关节角 -> 相对默认关节位置。"""
        q_real = np.asarray(q_real, dtype=np.float32)

        if q_real.shape != (6,):
            raise ValueError(
                f"Expected (6,), got {q_real.shape}"
            )

        return q_real - self.q_default

    def action_to_joint_target(self, action):
        """PPO Action -> 仿真定义的绝对关节目标。

        注意：只进行数学转换，不发送机器人命令。
        """
        action = np.asarray(action, dtype=np.float32)

        if action.shape != (6,):
            raise ValueError(
                f"Expected (6,), got {action.shape}"
            )

        if not np.isfinite(action).all():
            raise ValueError("Invalid action values")

        return self.q_default + self.action_scale * action
