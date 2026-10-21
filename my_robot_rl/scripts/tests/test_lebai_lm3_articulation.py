# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    python /home/l/isaac_space_01/my_robot_rl/scripts/tests/test_lebai_lm3_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="lebai_lm3 articulation test代码")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

#import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from my_robot_rl.assets.lebai_lm3 import LEBAI_LM3_CFG  # isort:skip


def design_scene() -> tuple[dict, list[float]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    origins = [0.0, 0.0, 0.0]    #这个origins原本是list[list[float]]，二维列表，单个机械臂被我改了记得类型注解应该与实际返回类型保持一致。

    # Articulation
    lebai_lm3_cfg = LEBAI_LM3_CFG.copy()
    lebai_lm3_cfg.prim_path = "/World/Origin/Robot"
    lebai_lm3 = Articulation(cfg=lebai_lm3_cfg)

    # return the scene information
    scene_entities = {"lebai_lm3": lebai_lm3}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["lebai_lm3"]

    # 使用调试模型断点调试，要注意看vscode左侧调试变量栏子，注意观察
    joint_one=robot.data.joint_names[0]
    print(f"[INFO]: joint_one的名字 : {joint_one}")  #就是lebai_lm3.py的joint_[1-6]  这里就打印个名
    
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop  不过下面的循环每次都是启动时 count=0，立即执行一次重置root（lebai中root就是base_link）,之后每经过 500 个仿真步，再执行一次。每次除了 root state，还重置 joint state 和内部缓冲状态
    # 还没有涉及joint_[1-6]，如果要设置其他6个关节的状态的话：
    """
    joint_1_pos = robot.data.joint_pos[:, 0].clone()  #这里只有一个机器人所以前面用了:,如果不只一个机器人的话记得加索引
    joint_1_vel = robot.data.joint_vel[:, 0].clone()
    print(f"joint_1_pos 1轴位置: {joint_1_pos}")   #位置是5.0979e-06，基本为0  弧度制
    print(f"joint_1_pos 1轴位置的shape: {joint_1_pos.shape}")  # shape=torch.Size([1])
    print(f"joint_1_vel 1轴位置: {joint_1_vel}")   #速度是0.0024,也基本为0吧就是说  角速度
    print(f"joint_1_vel 1轴位置的shape: {joint_1_vel.shape}")  # shape=torch.Size([1])

    # 如果你分不清楚名字可以自己命名一下：
    joint_names = [
        "shoulder_pan_joint",  #只是举例子的名字
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    indices = [robot.joint_names.index(name) for name in joint_names]
    six_axis_pos = robot.data.joint_pos[:, indices]  # (num_envs, 6)

    #如果你在 Reset 阶段想要直接重置这6个关节的状态，不能只克隆，还需要主动调用write_joint_state_to_sim
    # 假设你想把所有6个关节都设置为0
    joint_pos_target = torch.zeros(1, 6, device="cuda:0") # 形状要和关节数匹配
    joint_vel_target = torch.zeros(1, 6, device="cuda:0")
    robot.write_joint_state_to_sim(joint_pos_target, joint_vel_target)

    #如果你只是想让机械臂受控运动到某个位置，则需要在循环里不断调用目标设置：记得在循环里，就是下面的while里
    # 假设让1轴运动到0.5弧度
    target_pos = torch.zeros(1, 6, device="cuda:0")
    target_pos[:, 0] = 0.5
    robot.set_joint_position_target(target_pos)  #set_joint_position_target() 只是把目标写入内部缓冲区，并不直接将它应用到物理仿真，所以记得写入
    robot.write_data_to_sim()   #记得写入
    # 然后必须执行 sim.step() 让物理引擎计算
    sim.step()
    """

    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # 由于状态以仿真世界坐标系表示，因此我们将根状态相对于原点进行偏移。
            # 如果未执行此操作，机器人将被生成在模拟世界中的 (0, 0, 0) 处
            root_state = robot.data.default_root_state.clone()  #初始化状态数据，不是简单从usd复制状态
            
            
            print(f"[INFO]:root_state: {root_state.shape}")  #[1,13]
            print(f"[INFO]:root_state shape: {root_state}")
            

            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])  #写入base_link初始状态，都是0

            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()

            print(f"[INFO]:joint_pos: , {joint_pos.shape}")  #[1,6] 
            print(f"[INFO]:joint_vel: {joint_vel.shape}")  #[1,6]


            
            joint_pos += torch.rand_like(joint_pos) * 0.1   
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action
        # -- generate random joint efforts
        
        # 随机生成关节位置目标，如果不想要随机生成，可以使用固定的关节位置目标
        position = torch.randn_like(robot.data.joint_pos)
        # -- apply action to the robot
        robot.set_joint_position_target(position)
        # -- write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1

        """
        write_joint_state_to_sim()：直接设置关节的仿真位置和速度状态，可以理解为瞬间设置到指定状态，并非正常的运动过程。

        set_joint_position_target()：设置控制目标，配合 write_data_to_sim()、sim.step()，由配置好的关节驱动器与物理引擎使机械臂向目标运动。

        还有一个重要细节：B 不保证机械臂一定能到达目标。是否到达取决于驱动器、关节限制、负载和其他物理条件。
        """
        # Update buffers
        robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # 相机位置和朝向点
    sim.set_camera_view([2.5, 0.0, 3.0], [0.0, 0.0, 0.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()