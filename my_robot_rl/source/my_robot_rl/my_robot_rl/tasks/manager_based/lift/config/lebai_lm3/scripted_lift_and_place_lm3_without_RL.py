# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

   python /home/l/isaac_space_01/my_robot_rl/source/my_robot_rl/my_robot_rl/tasks/manager_based/lift/config/lebai_lm3/scripted_lift_and_place_lm3_without_RL.py

"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from collections.abc import Sequence

import warp as wp

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData

#import IsaacLab.source.isaaclab_tasks  # noqa: F401
from my_robot_rl.tasks.manager_based.lift.lift_env_cfg import LiftEnvCfg
#from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    PLACE_OBJECT = wp.constant(5)   # 放置后抬起
    RELEASE_OBJECT = wp.constant(6)  #末端保持在放置点，夹爪打开；单独计时，确保物体有足够时间释放。
    RETREAT = wp.constant(7) #夹爪保持打开，末端移动到放置点上方的安全位置。
    DONE = wp.constant(8)    #增加的完成状态


class PickSmWaitTime:
    """切换前状态的额外等待时间（单位为秒）."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.8)  # 原本是0.5
    APPROACH_OBJECT = wp.constant(0.6)    # 原本是0.6
    GRASP_OBJECT = wp.constant(1.0)   # 👈 这里就是闭合夹爪的等待时间（秒）原本是0.3秒
    LIFT_OBJECT = wp.constant(1.0)   # 原本是1.0秒
    PLACE_OBJECT = wp.constant(1.2)
    RELEASE_OBJECT = wp.constant(1.0)
    RETREAT = wp.constant(0.5)
    DONE = wp.constant(2.0)   #怎么都没进入这个DONE的状态，奇了怪了



@wp.func  #@wp.func：内核内部使用的辅助函数
# distance_below_threshold() 负责判断末端和目标的距离是否小于阈值。
# 你要能解释输入、输出和判断逻辑，不需要自己实现 Warp 的向量运算库
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel    #并行执行的入口
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),

    place_ee_pose: wp.array(dtype=wp.transform),  #加的放置点，11个数里面的倒数第三个哈，别对应错了

    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    # 获取线程ID
    tid = wp.tid()
    # 获取状态机状态
    state = sm_state[tid]
    # 定义每个状态
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # 等待一会儿
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # 进入下一个状态并重置等待时间
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_ABOVE_OBJECT:   #这里修正一下应该等待的是自己APPROACH_ABOVE_OBJECT设置的时间
                # 进入下一个状态并重置等待时间
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]   #APPROACH_OBJECT 当前直接将末端目标设为物体的根位姿，相当于末端目标与物体中心重合，这里刚好是因为物体是正方体，不同物体就会需要不同的位姿调整
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.PLACE_OBJECT
                sm_wait_time[tid] = 0.0  

    # 自己增加的部分  
    elif state == PickSmState.PLACE_OBJECT:
        # 1. 目标位置设为固定放置点
        des_ee_pose[tid] = place_ee_pose[tid]  
        # 2. 过程中保持夹紧 
        gripper_state[tid] = GripperState.CLOSE
        # 3. 判断：当前实际末端(ee_pose) 距离 放置点(place_ee_pose) 是否小于阈值
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,        #这里的threshold用的position_threshold=0.01，想改可以在main那里改
        ):
            # 4. 等待一会，进入下一个阶段
            if sm_wait_time[tid] >= PickSmWaitTime.PLACE_OBJECT:
                sm_state[tid] = PickSmState.RELEASE_OBJECT
                sm_wait_time[tid] = 0.0 

    elif state == PickSmState.RELEASE_OBJECT:
        # 松开夹爪
        des_ee_pose[tid] = place_ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.RELEASE_OBJECT:
            sm_state[tid] = PickSmState.RETREAT
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.RETREAT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):                     #这里的threshold用的position_threshold=0.01，想改可以在main那里改
            if sm_wait_time[tid] >= PickSmWaitTime.RETREAT:
                sm_state[tid] = PickSmState.DONE
                sm_wait_time[tid] = 0.0

    elif state == PickSmState.DONE:
        gripper_state[tid] = GripperState.CLOSE   #到达之后再关闭加爪
        if sm_wait_time[tid] >= PickSmWaitTime.DONE:   #用上设定给DONE的时间
            pass

    # 增加等待时间
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift and place an object.

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the 我设置放置位置的上方. 
    6. PLACE_OBJECT: 机械臂将物体方法到我设置的放置位置
    7. RELEASE_OBJECT: 释放物体
    8. RETREAT: 让机械臂回到放置位置的上方,这里我确保就是放置点的上方，我写为了固定点，在my_robot_rl/source/my_robot_rl/my_robot_rl/tasks/manager_based/lift/lift_env_cfg.py这个文件里
    7. DONE: 合并加爪
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        """初始化状态机

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)
        
        # 🟢 在这里直接写入固定的放置点
        # 暂时让末端在 (0.3, 0, 0.3)，姿态保持默认(单位四元数)
        # Warp: (x, y, z, qx, qy, qz, qw)
        # 初始化放置位姿
        self.des_place_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_place_ee_pose[:, 6] = 1.0  # 确保四元数合法 (w=1)   这里只初始化这个des_place_ee_pose，在下面计算那里再设置具体数值再传到warp

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """重置状态机"""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor,) -> torch.Tensor:
        """计算机器人末端执行器和夹具的期望状态"""
        # 将所有变换从 (w, x, y, z) 转换为 (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        object_quat = object_pose[:, 3:] 
        fixed_xyz = torch.tensor([0.3, -0.2, 0.1], device=self.device)
        self.des_place_ee_pose[:, 0:3] = fixed_xyz
        self.des_place_ee_pose[:, 3:] = object_quat
        
        # 转换到warp上
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # 4. 更新 warp
        self.des_place_ee_pose_wp = wp.from_torch(self.des_place_ee_pose.contiguous(), wp.transform)
        
        # 运行状态机
        wp.launch(      #wp.launch(dim=self.num_envs, ...) 会为每个环境启动对应的线程。wp.tid() 返回当前线程索引；在你的状态机中，它对应当前要处理的环境编号
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.des_place_ee_pose_wp,   #新加的放置点，11个数里面的倒数第三个哈，别对应错了
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # 将变换转换回 (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]   #小心下标别搞错了

        
        # 转换为torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    # 解析配置
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Lebai-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make("Isaac-Lift-Cube-Lebai-IK-Abs-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()

    """env 是套了外壳的“代理”，而 env.unwrapped 是剥掉外壳后，最原始的“本体”环境"""

    # 创建动作缓冲区（位置 + 四元数）
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    # 期望的物体姿态（我们仅对物体进行位置控制）
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # 创建状态机
    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device, position_threshold=0.01
    )

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]

            # observations
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- object frame
            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            # -- target object frame
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

            # advance state machine
            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),   #重新拼成[x,y,z,w,qx,qy,qz],注意这个dim=-1
                torch.cat([object_position, desired_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),
            )

            # # 在 main.py 的循环里，查看时间，特别是感觉哪个状态没执行可以这么看
            # for i, state in enumerate(pick_sm.sm_state):
            #     if state == PickSmState.RELEASE_OBJECT:
            #         print(f"Env {i} reached RELEASE_OBJECT. Wait time: {pick_sm.sm_wait_time[i]}")
            #         # 看看外部环境是不是在这时候触发了 结束

            # reset state machine
            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
