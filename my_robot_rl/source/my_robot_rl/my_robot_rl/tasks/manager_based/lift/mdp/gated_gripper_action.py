
import torch
from collections.abc import Sequence

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass


class GatedGripperAction(BinaryJointPositionAction):
    """LMG90 gripper action with distance gate and close latch."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        # Each environment has its own close latch.
        self.close_latched = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        # Debug data
        self.tcp_distance = torch.zeros(
            self.num_envs,
            device=self.device,
        )

    def process_actions(self, actions: torch.Tensor):
        # 1. Keep the original binary-action processing.
        super().process_actions(actions)

        # 2. Read TCP and cube positions in WORLD coordinates.
        tcp_pos_w = (
            self._env.scene["ee_frame"]
            .data.target_pos_w[:, 0, :]
        )

        cube_pos_w = (
            self._env.scene["object"]
            .data.root_pos_w[:, :3]
        )

        # 3. TCP-to-cube distance for each environment.
        self.tcp_distance = torch.linalg.vector_norm(
            tcp_pos_w - cube_pos_w,
            dim=-1,
        )

        # 4. PPO asks for CLOSE if action < 0.
        request_close = actions[:, 0] < 0.0

        # 5. Closing is allowed only near the cube.
        near_cube = (
            self.tcp_distance < self.cfg.close_distance
        )

        # 6. Latch CLOSE independently for each environment.
        self.close_latched |= (
            request_close & near_cube
        )

        # 7. Override processed actions.
        # True -> CLOSE (0 rad)
        # False -> OPEN (1 rad)
        self._processed_actions[:] = torch.where(
            self.close_latched.unsqueeze(-1),
            self._close_command.unsqueeze(0),
            self._open_command.unsqueeze(0),
        )

        # The existing task does not configure gripper clipping.
        # If clipping is added later, reapply it here.

    def reset(self, env_ids: Sequence[int] | None = None):
        # Reset original action buffers.
        super().reset(env_ids)

        if env_ids is None:
            self.close_latched.fill_(False)
            self._processed_actions[:] = self._open_command
        else:
            self.close_latched[env_ids] = False
            self._processed_actions[env_ids] = self._open_command


@configclass
class GatedGripperActionCfg(BinaryJointPositionActionCfg):
    """Configuration for distance-gated LMG90 control."""

    class_type: type[ActionTerm] = GatedGripperAction

    # Initial candidate; calibrate using successful scripted grasp.
    close_distance: float = 0.02
