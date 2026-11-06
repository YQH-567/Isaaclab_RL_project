#这个代码是专门用于计算机器人手指接触点的世界坐标位置的函数。它通过获取机器人指定部位的位姿（位置和四元数）以及局部偏移量，计算出接触点在世界坐标系中的位置。
#具体的位置你需要打开usd查看
import torch
from isaaclab.utils.math import quat_apply


# LMG90 grasp point local offsets (meters)自己去usd里加坐标系然后看transform
LEFT_GRASP_OFFSET = (0.00883, 0.02054, 0.0)
RIGHT_GRASP_OFFSET = (0.00914, 0.01951, 0.00013)


def get_finger_point_w(robot, body_id, offset_xyz):
    """Transform a finger grasp point from link frame to world frame."""

    link_pos_w = robot.data.body_pos_w[:, body_id, :]
    link_quat_w = robot.data.body_quat_w[:, body_id, :]

    offset = torch.as_tensor(
        offset_xyz,
        dtype=link_pos_w.dtype,
        device=link_pos_w.device,
    ).unsqueeze(0).expand_as(link_pos_w)

    return link_pos_w + quat_apply(link_quat_w, offset)