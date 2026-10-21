# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    python /home/l/isaac_space_01/my_robot_rl/scripts/tests/test_lebai_lm3_control.py

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

## 画误差图需要引进的包：
import matplotlib.pyplot as plt
import numpy as np   #GPU 张量无法直接用 matplotlib 画图，所以需要转回numpy数组
import torch


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
    robot = entities["lebai_lm3"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

# # 实验一：保持 LM3 默认姿态
#     while simulation_app.is_running():
#         # Reset
#         if count % 500 == 0:
#             # reset counter
#             count = 0
#             robot.reset()  #基本都会恢复初始状态，就是我原本usd设置的状态
#             print("[INFO]: Resetting robot state...")
#         robot.write_data_to_sim()
#         # Perform step
#         sim.step()
#         # Increment counter
#         count += 1
#         # Update buffers
#         robot.update(sim_dt)

# #实验二： 只让关节 1 转动 1.57 rad，转个90度
#     while simulation_app.is_running():
#         # Reset
#         if count % 500 == 0:
#             # reset counter
#             count = 0
#             # reset the scene entities
#             # root state
#             # 由于状态以仿真世界坐标系表示，因此我们将根状态相对于原点进行偏移。
#             # 如果未执行此操作，机器人将被生成在模拟世界中的 (0, 0, 0) 处
#             # 并且这次实验可不能把下面到robot.reset()的代码给删掉了，不然第一次到90度之后就一直在90了，因为没有在这个if里面转回原来的状态
#             root_state = robot.data.default_root_state.clone()  #初始化状态数据，不是简单从usd复制状态
#             root_state[:, :3] += origins
#             robot.write_root_pose_to_sim(root_state[:, :7])
#             robot.write_root_velocity_to_sim(root_state[:, 7:])  #写入base_link初始状态，都是0

#             # set joint positions with some noise
#             joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
#             robot.write_joint_state_to_sim(joint_pos, joint_vel)

#             # clear internal buffers
#             robot.reset()
#             print("[INFO]: Resetting robot state...")

#         joint_target_pos=robot.data.default_joint_pos.clone()
#         #print(f"1轴的初始位置： {joint_1_default_pos}")
#         joint_target_pos[:,0] += 1.57
#         # 设置目标并推入物理引擎
#         robot.set_joint_position_target(joint_target_pos)
#         robot.write_data_to_sim()    
        
#         # Perform step
#         sim.step()
#         # Increment counter
#         count += 1
#         # Update buffers
#         robot.update(sim_dt)

# 实验三：记录关节位置误差
    error_history=[]
    while simulation_app.is_running():
        # Reset重置部分
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # 由于状态以仿真世界坐标系表示，因此我们将根状态相对于原点进行偏移。
            # 如果未执行此操作，机器人将被生成在模拟世界中的 (0, 0, 0) 处
            root_state = robot.data.default_root_state.clone()  #初始化状态数据，不是简单从usd复制状态
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])  #写入base_link初始状态，都是0

            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)   #注意区别下面的write_data_to_sim()
            robot.reset()

        #记录轴位置误差
        # 1.设定目标位置（比如让轴加0.5弧度）
        joint_target_pos=robot.data.default_joint_pos.clone()
        joint_target_pos += -0.5
        # 2.设置目标并推入物理引擎
        robot.set_joint_position_target(joint_target_pos)
        robot.write_data_to_sim()    
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)

        error = joint_target_pos-robot.data.joint_pos
        error_np = error.detach().cpu().numpy()
        error_history.append(error_np.copy())
        if count >= 500:
            error_array = np.array(error_history)
            print(f"error_history的数据结构我看看咋样： {error_array.shape}")   #奥(500, 1, 6)
            break
        
    # 把列表转换为 numpy 数组，形状变为(500, 1, 6)
    error_array = np.array(error_history)
    # 创建图表
    plt.figure(figsize=(10, 6))
    x = np.arange(error_array.shape[0])  # [0, 1, 2, ..., 499]

    # 画出6个关节的误差曲线，不过注意上面我只改了一个轴的
    for i in range(6):
        y_joint_i = error_array[:, 0, i]
        plt.plot(x,y_joint_i, label=f'Joint {i+1}')

    plt.title('Joint Position Error over Time')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Error (rad)')
    plt.legend()
    plt.grid(True)
    plt.show() # 注意：这行代码会阻塞，直到你手动关闭图片窗口


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

    robot = scene_entities["lebai_lm3"]

    print("Joint names:", robot.joint_names)
    print("Stiffness:", robot.data.joint_stiffness)
    print("Damping:", robot.data.joint_damping)
    print("Effort limits:", robot.data.joint_effort_limits)
    print("Velocity limits:", robot.data.joint_vel_limits)
    print("Position limits:", robot.data.joint_pos_limits)
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()