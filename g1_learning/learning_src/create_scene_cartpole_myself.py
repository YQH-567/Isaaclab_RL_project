"""
这是cartpole的示例脚本，显示描述2个cartpole的运动，理解这个关节控制的概念，进而理解32个cartpole的并行理念，为后续做个铺垫
这里的启动方式是： python /home/l/isaac_space_01/g1_learning/learning_src/create_scene_cartpole_myself.py
"""

"""
这里需要理解的核心的地方是：
1. articulation:
    多个刚体
    +
    关节连接
    +
    动力学约束
"""

import argparse

from isaaclab.app import AppLauncher

# 添加参数
parser = argparse.ArgumentParser(description="使用交互式场景界面的教程,核心是理解关节运动")
parser.add_argument("--num_envs",type=int,default=2,help="生成环境的数量，可以自定义，默认是2个")
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 加载app
app_launcher = AppLauncher(args_cli)  #AppLauncher 对象
simulation_app = app_launcher.app

"其余代码如下："
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  #导入Articulation配置和基础Asset配置
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass  #这里导入

from isaaclab_assets import CARTPOLE_CFG  #这里导入cartpole 的配置

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """做一个cart-pole 的场景配置"""

    #地面配置
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane",spawn=sim_utils.GroundPlaneCfg())
    # 灯光
    demo_light = AssetBaseCfg(prim_path="/World/Light",spawn=sim_utils.DomeLightCfg(intensity=3000.0,color=(0.75,0.75,0.75)))

    #关节
    cartpole: ArticulationCfg=CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def run_simulator(sim: sim_utils.SimulationContext,scene: InteractiveScene):
    """仿真循环"""
    #   提取场景实体
    # 官方案例是为了可读性？
    robot = scene["cartpole"]     #这里表示从InteractiveScene调的cartpole
    # 定义仿真歩进
    sim_dt = sim.get_physics_dt()
    count = 0
    # 仿真循环
    while simulation_app.is_running():
        # 500步重置
        if count % 500 == 0:
            count = 0 
            # 重置场景实体
            # 根状态
            # 由于状态是在仿真世界框架中编写的，所以我们用原点抵消了根状态
            # 如果不这样做，那么机器人将在模拟世界的（0,0,0）处生成
            root_state = robot.data.default_root_state.clone()  #这里的表示根状态和机器人默认根状态一样，这里可以打开articulation_data.py查看shape,查看如下：
            """Default root state ``[pos, quat, lin_vel, ang_vel]`` in the local environment frame. Shape is (num_instances, 13).'3+4+3+3=13'"""
            
            root_state[:,:3] += scene.env_origins    #这里的env_origins的shape是(num_envs, 3)，第一维是环境数量，第二维是xyz
            robot.write_root_pose_to_sim(root_state[:, :7])  # pos和quat
            robot.write_root_velocity_to_sim(root_state[:, 7:]) # lin_vel,ang_vel
            # 用一些噪音设定关节位置
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(),robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos,joint_vel)   #直接把当前 joint position 和 joint velocity 写进 simulation state。
            # 清理内部缓存
            scene.reset()  #主要是对 scene 中各 entity 的内部 buffer 做 reset
            print("[INFO]: Resetting robot state...")

        # 用随机动作
        #   生成随机的联合工作
        # 生成一个与 joint_pos 相同 shape 的标准正态随机 Tensor，
        # shape = (num_envs, num_joints)，并把随机 effort 的幅值扩大 5 倍
        
        # 实验0,原来的官方代码
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0  #或者这么说生成的是均值为 0、标准差为 5 的正态分布随机关节力矩/力，形状和 joint_pos 一样
        
        #小实验1 ，让slider_to_cart全为5
        # print("joint_name:", robot.joint_names)
        # print("joint_pose shape: ",robot.data.joint_pos.shape)
        # cart_ids, _=robot.find_joints("slider_to_cart")
        # pole_ids, _=robot.find_joints("cart_to_pole")
        # efforts = torch.zeros_like(robot.data.joint_pos)   #全是0
        # efforts[:,cart_ids]=5.0  # 全设置为5，且只控制slider_to_cart

        #小实验2,让cart_to_pole全是1.0
        # print("joint_name:", robot.joint_names)
        # print("joint_pose shape: ",robot.data.joint_pos.shape)  #[32个env,[slider_to_cart,cart_to_pole]]
        # cart_ids, _=robot.find_joints("slider_to_cart")
        # pole_ids, _=robot.find_joints("cart_to_pole")
        # efforts = torch.zeros_like(robot.data.joint_pos)   #全是0
        # efforts[:,pole_ids]=1.0  # 全设置为1，且只控制cart_to_pole   

        #小实验3  只控制第0个 environment 的 cart 并令值为5.0   
        # print("joint_name:", robot.joint_names)
        # print("joint_pose shape: ",robot.data.joint_pos.shape)  #[32个env,[slider_to_cart,cart_to_pole]]
        # cart_ids, _=robot.find_joints("slider_to_cart")
        # pole_ids, _=robot.find_joints("cart_to_pole")
        # efforts = torch.zeros_like(robot.data.joint_pos)   #全是0
        # efforts[0:1,cart_ids]=5.0  # 全设置为1，且只控制cart_to_pole   
        # efforts[1:32,cart_ids]=0.0

        # #小实验4 列表拼接一下：
        # print("joint_name:", robot.joint_names)
        # print("joint_pose shape: ",robot.data.joint_pos.shape)  #[32个env,[slider_to_cart,cart_to_pole]]
        # cart_ids, _=robot.find_joints("slider_to_cart")
        # pole_ids, _=robot.find_joints("cart_to_pole")
        # efforts = torch.zeros_like(robot.data.joint_pos)   #全是0
        # #efforts[:,cart_ids+pole_ids]=5.0    #全是5，两个是tensor,直接加容易语义误导
        # joint_ids = cart_ids + pole_ids
        # efforts[:, joint_ids] = 5.0

        #   执行机器人动作
        robot.set_joint_effort_target(efforts)
        #   记入仿真数据
        
        scene.write_data_to_sim()
        sim.step()
        count +=1
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # 设置相机
    sim.set_camera_view([2.5,0.0,4.0],[0.0,0.0,2.0])
    # 设计场景
    scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs,env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)

if __name__== "__main__":
    main()
    simulation_app.close()
