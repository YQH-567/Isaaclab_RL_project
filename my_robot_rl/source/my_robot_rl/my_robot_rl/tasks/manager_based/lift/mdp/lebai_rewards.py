
import torch
from isaaclab.managers import SceneEntityCfg
from .geometry import (
    LEFT_GRASP_OFFSET,
    RIGHT_GRASP_OFFSET,
    get_finger_point_w,
)







import torch
from isaaclab.utils.math import subtract_frame_transforms


def lebai_goal_tracking(
    env,
    command_name="object_pose",
    std=0.15,
    min_lift=0.02,
    rest_z=0.021,
):
    robot = env.scene["robot"]
    obj = env.scene["object"]

    # Convert cube position from world frame
    # to robot root frame.
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        obj.data.root_pos_w,
    )

    # Target object position in robot root frame.
    target_pos_b = env.command_manager.get_command(
        command_name
    )[:, :3]

    distance = torch.linalg.vector_norm(
        object_pos_b - target_pos_b,
        dim=-1,
    )

    # Relative lifting height above reset position.
    object_z = (
        obj.data.root_pos_w[:, 2]
        - env.scene.env_origins[:, 2]
    )

    lift_height = object_z - rest_z
    lifted = lift_height > min_lift

    # Only reward target tracking after lifting.
    return lifted.float() * torch.exp(-distance / std)


def _get_object_tcp(env, object_cfg):
    """Get object position and the same TCP frame used by scripted lift."""

    obj = env.scene[object_cfg.name]
    ee_sensor = env.scene["ee_frame"]

    object_pos_w = obj.data.root_pos_w[:, :3]
    tcp_pos_w = ee_sensor.data.target_pos_w[:, 0, :]

    distance = torch.linalg.vector_norm(
        tcp_pos_w - object_pos_w, dim=-1
    )

    return obj, object_pos_w, tcp_pos_w, distance


def _is_gripper_closed(env):
    """Binary gripper: processed command 0=close, 1=open."""

    action = env.action_manager.get_term(
        "gripper_action"
    ).processed_actions

    return action.reshape(env.num_envs, -1)[:, 0] < 0.5


def lebai_tcp_object_distance(
    env,
    std=0.15,
    object_cfg=SceneEntityCfg("object"),
):
    """Dense TCP-object reaching reward."""

    _, _, _, distance = _get_object_tcp(env, object_cfg)

    #return 1.0 - torch.tanh(distance / std)
    return torch.exp(-distance / std)  #恢复足够宽的梯度范围


def lebai_finger_enclosure(
    env,
    left_tip_cfg: SceneEntityCfg,
    right_tip_cfg: SceneEntityCfg,
    center_std=0.035,
    symmetry_std=0.025,
    target_gap=0.055,
    gap_std=0.035,
):
    """Reward symmetric, opposing fingertip geometry around the cube."""

    robot = env.scene["robot"]
    obj = env.scene["object"]

    left_pos = get_finger_point_w(
        robot,
        left_tip_cfg.body_ids[0],
        LEFT_GRASP_OFFSET,
    )

    right_pos = get_finger_point_w(
        robot,
        right_tip_cfg.body_ids[0],
        RIGHT_GRASP_OFFSET,
    )

    object_pos = obj.data.root_pos_w[:, :3]

    v_left = left_pos - object_pos
    v_right = right_pos - object_pos

    d_left = torch.linalg.vector_norm(
        v_left, dim=-1
    ).clamp_min(1e-6)

    d_right = torch.linalg.vector_norm(
        v_right, dim=-1
    ).clamp_min(1e-6)

    # Fingers should approach from opposite sides.
    cosine = torch.sum(v_left * v_right, dim=-1) / (
        d_left * d_right
    )
    cosine = torch.clamp(cosine, -1.0, 1.0)

    opposite = 0.5 * (1.0 - cosine)

    # Symmetric distances from both fingers to the cube.
    symmetry = torch.exp(
        -((d_left - d_right) / symmetry_std) ** 2
    )

    # Cube should be centered between the fingertips.
    midpoint = 0.5 * (left_pos + right_pos)

    center_error = torch.linalg.vector_norm(
        midpoint - object_pos, dim=-1
    )


    # Initial aperture shaping.
    gap = torch.linalg.vector_norm(
        left_pos - right_pos, dim=-1
    )

    #return opposite * symmetry * centered * gap_score  
    # 这里暂时取消这个gap_score的影响，因为在实际操作中，手指的开合距离可能会受到物体大小和形状的影响，过于严格的gap约束可能会限制机器人的灵活性。
    #return opposite * symmetry * centered  #这个奖励数值太小了
    #暂时用这个试试，只有中心距离较近时，指尖的对侧关系和对称性才会贡献更多奖励
    center_score = torch.exp(
    -center_error / 0.15)

    return center_score * (
        0.65
        + 0.25 * opposite
        + 0.10 * symmetry
    )

def lebai_gripper_open_enclosure(
    env,
    left_tip_cfg,
    right_tip_cfg,
):
    """Encourage keeping fingers open while positioning around the cube."""

    enclosure = lebai_finger_enclosure(
        env, left_tip_cfg, right_tip_cfg
    )

    return enclosure * (~_is_gripper_closed(env)).float()


def lebai_gripper_close_when_enclosed(
    env,
    left_tip_cfg,
    right_tip_cfg,
):
    """Reward closing only when geometric enclosure is good."""

    enclosure = lebai_finger_enclosure(
        env, left_tip_cfg, right_tip_cfg
    )

    return enclosure * _is_gripper_closed(env).float()


def lebai_premature_close(
    env,
    left_tip_cfg,
    right_tip_cfg,
):
    """Penalize closing before the cube is properly enclosed."""

    enclosure = lebai_finger_enclosure(
        env, left_tip_cfg, right_tip_cfg
    )

    return (
        (1.0 - enclosure)
        * _is_gripper_closed(env).float()
    )


def _contact_force(env, sensor_name):
    """Filtered normal contact force between one finger and the cube."""

    sensor = env.scene[sensor_name]

    force_matrix = sensor.data.force_matrix_w

    force = torch.linalg.vector_norm(
        force_matrix, dim=-1
    )

    return force.sum(dim=(1, 2))


def _grasp_contacts(
    env,
    left_sensor_name="left_finger_contact",
    right_sensor_name="right_finger_contact",
    threshold=0.5,
):
    left_force = _contact_force(env, left_sensor_name)
    right_force = _contact_force(env, right_sensor_name)

    left_contact = left_force > threshold
    right_contact = right_force > threshold

    return left_force, right_force, left_contact, right_contact


def lebai_bilateral_grasp(
    env,
    left_sensor_name="left_finger_contact",
    right_sensor_name="right_finger_contact",
    threshold=0.5,
):
    """Reward simultaneous, reasonably balanced bilateral contact."""

    left_force, right_force, left, right = _grasp_contacts(
        env, left_sensor_name, right_sensor_name, threshold
    )

    bilateral = left & right

    force_balance = 1.0 - torch.abs(
        left_force - right_force
    ) / (left_force + right_force + 1e-6)

    return bilateral.float() * torch.clamp(
        force_balance, 0.0, 1.0
    )


def lebai_single_contact_penalty(
    env,
    left_sensor_name="left_finger_contact",
    right_sensor_name="right_finger_contact",
    threshold=0.5,
):
    """Penalize closing with only one finger contacting the cube."""

    _, _, left, right = _grasp_contacts(
        env, left_sensor_name, right_sensor_name, threshold
    )

    single_contact = left ^ right

    return (
        single_contact.float()
        * _is_gripper_closed(env).float()
    )


def _lift_delta(obj, env, rest_z=0.055):
    """Object lift relative to its resting height, not world origin."""

    object_z_local = (
        obj.data.root_pos_w[:, 2]
        - env.scene.env_origins[:, 2]
    )

    return torch.clamp(
        object_z_local - rest_z, min=0.0
    )


def lebai_lift_progress(
    env,
    rest_z=0.055,
    target_lift=0.10,
    left_sensor_name="left_finger_contact",
    right_sensor_name="right_finger_contact",
    contact_threshold=0.5,
    object_cfg=SceneEntityCfg("object"),
):
    """Dense lift progress, enabled only during a real bilateral grasp."""

    obj, _, _, tcp_distance = _get_object_tcp(
        env, object_cfg
    )

    _, _, left, right = _grasp_contacts(
        env, left_sensor_name, right_sensor_name,
        contact_threshold
    )

    bilateral = left & right
    closed = _is_gripper_closed(env)

    lift_height = _lift_delta(obj, env, rest_z)

    carry_gate = (
        bilateral
        #& closed
        & (tcp_distance < 0.08)
    )

    progress = torch.clamp(
        lift_height / target_lift, 0.0, 1.0
    )

    return progress * carry_gate.float()


def lebai_stable_lift(
    env,
    rest_z=0.055,
    min_lift=0.04,
    max_object_speed=0.15,
    left_sensor_name="left_finger_contact",
    right_sensor_name="right_finger_contact",
    contact_threshold=0.5,
    object_cfg=SceneEntityCfg("object"),
):
    """Success signal for a stable, bilateral object lift."""

    obj, _, _, tcp_distance = _get_object_tcp(
        env, object_cfg
    )

    _, _, left, right = _grasp_contacts(
        env, left_sensor_name, right_sensor_name,
        contact_threshold
    )

    bilateral = left & right
    closed = _is_gripper_closed(env)

    lift_height = _lift_delta(obj, env, rest_z)

    object_speed = torch.linalg.vector_norm(
        obj.data.root_lin_vel_w, dim=-1
    )

    stable = (
        (lift_height > min_lift)
        & bilateral
        # & closed
        & (tcp_distance < 0.08)
        & (object_speed < max_object_speed)
    )

    return stable.float()


def lebai_object_goal_distance(
    env,
    std=0.3,
    min_lift=0.04,
    rest_z=0.055,
    command_name="object_pose",
    left_sensor_name="left_finger_contact",
    right_sensor_name="right_finger_contact",
    contact_threshold=0.5,
    object_cfg=SceneEntityCfg("object"),
):
    """Track the object target only after bilateral grasp and lift."""

    obj = env.scene[object_cfg.name]

    _, _, left, right = _grasp_contacts(
        env, left_sensor_name, right_sensor_name,
        contact_threshold
    )

    lift_height = _lift_delta(obj, env, rest_z)

    bilateral = left & right
    closed = _is_gripper_closed(env)

    object_pos_local = (
        obj.data.root_pos_w[:, :3]
        - env.scene.env_origins
    )

    command = env.command_manager.get_command(command_name)

    distance = torch.linalg.vector_norm(
        object_pos_local - command[:, :3],
        dim=-1
    )

    gate = (
        bilateral
        & closed
        & (lift_height > min_lift)
    )

    return (
        1.0 - torch.tanh(distance / std)
    ) * gate.float()


def lebai_arm_action_rate_l2(env):
    """Penalize only the six arm actions, not the binary gripper switch."""

    action = env.action_manager.action[:, :6]
    previous_action = env.action_manager.prev_action[:, :6]

    return torch.sum(
        (action - previous_action) ** 2,
        dim=-1
    )
