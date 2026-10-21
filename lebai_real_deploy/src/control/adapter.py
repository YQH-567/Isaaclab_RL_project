
# import numpy as np


# class LM3Adapter:
#     def __init__(self):
#         # Isaac Lab 训练时的默认关节位置
#         self.q_default = np.array(
#             [3.13, -1.97, 1.71, -1.29, -1.62, 0.0],
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


# # 验证位置之后验证速度：

#     def joint_velocity_observation(self, qd_real):
#         """六轴实际速度 -> joint_vel_rel。"""
#         qd_real = np.asarray(qd_real, dtype=np.float32)

#         if qd_real.shape != (6,):
#             raise ValueError(
#                 f"Expected 6 joint velocities, got {qd_real.shape}"
#             )

#         if not np.isfinite(qd_real).all():
#             raise ValueError("Invalid joint velocities")

#         # 当前训练环境的默认关节速度为零
#         return qd_real

#     def get_proprioception(self, kin_data):
#         """从 SDK 状态构造前 12 维本体观测。"""
#         q_real = kin_data["actual_joint_pose"]
#         qd_real = kin_data["actual_joint_speed"]

#         joint_pos_rel = self.joint_position_observation(q_real)
#         joint_vel_rel = self.joint_velocity_observation(qd_real)

#         proprioception = np.concatenate(
#             [joint_pos_rel, joint_vel_rel]
#         ).astype(np.float32)

#         assert proprioception.shape == (12,)
#         return proprioception



import numpy as np


class LM3Adapter:
    def __init__(self):
        # 与训练时的 LEBAI_LM3_CFG 一致
        self.q_default = np.array(
            [3.13, -1.97, 1.71, -1.29, -1.62, 0.0],
            dtype=np.float32,
        )

        # JointPositionActionCfg
        self.action_scale = 0.5

        # 上一轮策略原始动作
        self.last_action = np.zeros(6, dtype=np.float32)

    @staticmethod
    def check_vector(value, size, name):
        value = np.asarray(value, dtype=np.float32)

        if value.shape != (size,):
            raise ValueError(
                f"{name}: expected ({size},), got {value.shape}"
            )

        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")

        return value

    def joint_position_observation(self, q_real):
        q_real = self.check_vector(q_real, 6, "joint_position")
        return q_real - self.q_default

    def joint_velocity_observation(self, qd_real):
        qd_real = self.check_vector(qd_real, 6, "joint_velocity")

        # 训练默认速度为零
        return qd_real

    def get_proprioception(self, kin_data):
        pos = self.joint_position_observation(
            kin_data["actual_joint_pose"]
        )

        vel = self.joint_velocity_observation(
            kin_data["actual_joint_speed"]
        )

        return np.concatenate([pos, vel]).astype(np.float32)

    def build_observation(self, kin_data, target_pose):
        """
        构造 PPO 的 25 维原始 Observation。

        此处不做归一化：
        RunningStandardScaler 已封装在 TorchScript 中。
        """
        proprio = self.get_proprioception(kin_data)

        target_pose = self.check_vector(
            target_pose, 7, "target_pose"
        )

        # 目标四元数规范化检查
        quat_norm = np.linalg.norm(target_pose[3:7])
        if not np.isclose(quat_norm, 1.0, atol=1e-4):
            raise ValueError(
                f"Invalid target quaternion norm: {quat_norm}"
            )

        obs = np.concatenate([
            proprio,             # 0:12
            target_pose,          # 12:19
            self.last_action,     # 19:25
        ]).astype(np.float32)

        assert obs.shape == (25,)
        return obs

    def action_to_joint_target(self, action):
        """
        仅复现动作的数学映射。
        不包含真实机器人的运动安全判断。
        """
        action = self.check_vector(action, 6, "action")
        return self.q_default + self.action_scale * action

    def update_last_action(self, action):
        self.last_action = self.check_vector(
            action, 6, "action"
        ).copy()

    def reset(self):
        self.last_action[:] = 0.0

"""
这里最需要理解的部分：

obs = np.concatenate([
    proprio,
    target_pose,
    self.last_action,
])

它把我们分开测试的四组数据按照训练定义连接起来，最终成为 (25,)。

而模型调用时会加上一个 batch 维度，变成 (1, 25)。
"""