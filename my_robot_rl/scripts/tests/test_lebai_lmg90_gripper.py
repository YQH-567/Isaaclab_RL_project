"""
使用：

"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="lmg90_gripper articulation test 代码")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

#from my_robot_rl.source.my_robot_rl.my_robot_rl.assets.lebai_lm3 import LMG90_GRIPPER_CFG
from my_robot_rl.assets.lebai_lm3_lmg90 import LEBAI_LM3_LMG90_CFG


## 画误差图需要引进的包：
import matplotlib.pyplot as plt
import numpy as np   #GPU 张量无法直接用 matplotlib 画图，所以需要转回numpy数组
import torch

def design_scene() -> tuple[dict,list[float]]:
    """场地设置"""
    #地面
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane",cfg)
    #灯光
    cfg = sim_utils.DomeLightCfg(intensity=3000.0,color=(0.75,0.75,0.75))
    cfg.func("/World/Light",cfg)
    origins = [0.0,0.0,0.0]

    #设置关节
    lebai_lm3_lmg90_gripper_cfg = LEBAI_LM3_LMG90_CFG.copy()
    lebai_lm3_lmg90_gripper_cfg.prim_path = "/World/Origin/Robot"
    lebai_lm3_lmg90_gripper = Articulation(lebai_lm3_lmg90_gripper_cfg)

    scene_entities = {"lebai_lm3_lmg90_gripper":lebai_lm3_lmg90_gripper}
    return scene_entities,origins

def run_simulator(sim:sim_utils.SimulationContext,entities: dict[str,Articulation],origins:torch.Tensor):
    """设置reset"""

    robot = entities["lebai_lm3_lmg90_gripper"]
    sim_dt = sim.get_physics_dt()
    count = 0
    error_history=[]
    error_history_gripper=[]

    while simulation_app.is_running():
        #设置reset
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            #print(f"让我看看这个root_state有没有搞鬼: {root_state}")  #并没有搞鬼
            root_state[:,:3] += origins
            robot.write_root_pose_to_sim(root_state[:,:7])
            robot.write_root_velocity_to_sim(root_state[:,7:])

            joint_pos,joint_vel = robot.data.default_joint_pos.clone(),robot.data.default_joint_vel.clone()
            #print(f"让我看看这个joint_pos有没有搞鬼: {joint_pos}")  #也没有搞鬼tensor([[ 3.1300, -1.9700,  1.7100, -1.2900, -1.6200,  0.0000,  0.0000,  0.0000,
                                                                #  0.0000,  0.0000,  0.0000,  0.0000]], device='cuda:0')
            robot.write_joint_state_to_sim(joint_pos,joint_vel)
            robot.reset()
            print("[INFO]:重置lmg90_gripper")
            # 关键：给 PD 一个目标
            robot.set_joint_position_target(joint_pos)

        # joint_pos,joint_vel = robot.data.default_joint_pos.clone(),robot.data.default_joint_vel.clone()
        # robot.write_joint_state_to_sim(joint_pos,joint_vel)  #有这个就不会回0点，但是按道理没有这个不也应该在0点吗
        # 好了我懂了，因为执行器是往0那里，所以需要reset之后设置执行器目标：robot.set_joint_position_target(joint_pos)

        joint_3_id=robot.joint_names.index("joint_2")
        gripper_r_joint1_id=robot.joint_names.index("gripper_r_joint1")
        joint_pos=robot.data.default_joint_pos.clone()
        #gripper_r_joint1_pos=robot.data.default_joint_pos.clone()
        joint_pos[:,joint_3_id] += 0.3
        joint_pos[:,gripper_r_joint1_id] += 0.3
        robot.set_joint_position_target(joint_pos)
        #robot.set_joint_position_target(gripper_r_joint1_pos)
        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)

        error = joint_pos[:, joint_3_id] - robot.data.joint_pos[:, joint_3_id]
        error_np = error.detach().cpu().numpy()
        error_history.append(error_np.copy())

        error_gripper = joint_pos[:, gripper_r_joint1_id] - robot.data.joint_pos[:, gripper_r_joint1_id]
        error_np_gripper = error_gripper.detach().cpu().numpy()
        error_history_gripper.append(error_np_gripper.copy())

        if count >= 500:
            error_array = np.array(error_history)
            error_array_gripper = np.array(error_history_gripper)
            break
    #
    #画图部分
    error_array = np.array(error_history)
    error_array_gripper = np.array(error_history_gripper)

    # 创建画布：2行1列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # ========== 第一张图：机械臂第三个轴 (索引为 2) ==========
    # 假设 error_array 形状为 (steps, num_envs, 6)，取所有步、第0个环境、第3个关节(索引2)
    x1 = np.arange(error_array.shape[0])
    ax1.plot(x1, error_array[:, 0], label="Joint 3 (Index 2)", color='blue')
    
    ax1.set_title("LM3 Drive Joint 3 Position Error")
    ax1.set_xlabel("Simulation Steps")
    ax1.set_ylabel("Position Error (rad)")
    ax1.legend()
    ax1.grid(True)

    # ========== 第二张图：夹爪 (gripper_r_joint1) ==========
    x2 = np.arange(error_array_gripper.shape[0])
    # 沿用你原来的写法
    ax2.plot(x2, error_array_gripper[:,0], label="gripper_r_joint1", color='orange')

    ax2.set_title("LMG90 Gripper Joint Position Error")
    ax2.set_xlabel("Simulation Steps")
    ax2.set_ylabel("Position Error (rad)")
    ax2.legend()
    ax2.grid(True)

    # 自动调整间距
    plt.tight_layout()
    plt.show()

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    #设置像机
    sim.set_camera_view([2.0,2.0,2.0],[0.0,0.0,0.0])
    scene_entities,scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins,device=sim.device)
    sim.reset()
    print("[INFO]:Setup complete...")
    run_simulator(sim,scene_entities,scene_origins)

if __name__ == "__main__":
    main()
    simulation_app.close()