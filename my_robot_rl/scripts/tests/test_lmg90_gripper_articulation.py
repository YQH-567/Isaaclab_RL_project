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
from my_robot_rl.assets.lmg90_gripper import LMG90_GRIPPER_CFG


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
    lmg90_gripper_cfg = LMG90_GRIPPER_CFG.copy()
    lmg90_gripper_cfg.prim_path = "/World/Origin/Robot"
    lmg90_gripper = Articulation(lmg90_gripper_cfg)

    scene_entities = {"lmg90_gripper":lmg90_gripper}
    return scene_entities,origins

def run_simulator(sim:sim_utils.SimulationContext,entities: dict[str,Articulation],origins:torch.Tensor):
    """设置reset"""

    robot = entities["lmg90_gripper"]
    sim_dt = sim.get_physics_dt()
    count = 0
    error_history=[]    

    while simulation_app.is_running():
        #设置reset
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()

            root_state[:,:3] += origins
            robot.write_root_pose_to_sim(root_state[:,:7])
            robot.write_root_velocity_to_sim(root_state[:,7:])

            joint_pos,joint_vel = robot.data.default_joint_pos.clone(),robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos,joint_vel)
            print(f"看看数据格式： {joint_pos.shape}")
            robot.reset()
            print("[INFO]:重置lmg90_gripper")

        joint_target_pos=robot.data.default_joint_pos.clone()
        index_id=robot.joint_names.index("gripper_r_joint1")
        joint_target_pos[:,index_id]+=0.3
        #print(f"我看看第几个关节名才是动力关节： {robot.data.joint_names}")  #好了我发现是第3个
        robot.set_joint_position_target(joint_target_pos)
        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)

        error = joint_target_pos[:, index_id] - robot.data.joint_pos[:, index_id]
        error_np = error.detach().cpu().numpy()
        error_history.append(error_np.copy())
        if count >= 500:
            error_array = np.array(error_history)
            print(f"error_history的数据结构我看看咋样： {error_array.shape}")   #奥(500, 1, 6)
            break

    error_array = np.array(error_history)

    plt.figure(figsize=(10, 6))
    x = np.arange(error_array.shape[0])

    plt.plot(x, error_array[:, 0], label="gripper_r_joint1")

    plt.title("LMG90 Drive Joint Position Error")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Position Error (rad)")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    #设置像机
    sim.set_camera_view([0.5,0.5,0.5],[0.0,0.0,0.0])
    scene_entities,scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins,device=sim.device)
    sim.reset()
    print("[INFO]:Setup complete...")
    run_simulator(sim,scene_entities,scene_origins)

if __name__ == "__main__":
    main()
    simulation_app.close()