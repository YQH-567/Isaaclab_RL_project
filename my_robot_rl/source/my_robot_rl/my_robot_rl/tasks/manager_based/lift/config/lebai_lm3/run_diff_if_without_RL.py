"""
参考官方微分IK代码：
./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py

本代码使用方式： 
python /home/l/isaac_space_01/my_robot_rl/source/my_robot_rl/my_robot_rl/tasks/manager_based/lift/config/lebai_lm3/run_diff_if_without_RL.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="乐白机械臂微分IK")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers   #这个肯定是显示xyz方向
from isaaclab.markers.config import FRAME_MARKER_CFG   #这个marker是干啥的，画那三个箭头？奥就是那三个箭头，但我感觉直接在这写好了，或者我先直接用好了，好像没什么特别的地方，ok就是没啥特别的，用这个也行
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

##
# 定义好配置
##
from my_robot_rl.assets.lebai_lm3_lmg90 import LEBAI_LM3_LMG90_CFG

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):    #这里跟def design_scene() -> tuple[dict,list[float]]:有啥区别？
    #奥区别是这样的： 前者是声明式场景配置，适合 InteractiveScene、多环境复制和 Manager RL；你以前的 design_scene() 是直接 imperatively spawn 对象
    """Configuration for a cart-pole scene."""

    # 设置地面
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # 设置灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # 底座设置
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # 关节电机设置
    robot = LEBAI_LM3_LMG90_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")   
    """
    这个replace的作用:
    基于默认的 Lebai 机械臂配置，派生出一个全新的、完全独立的配置副本，并将其在物理场景中的挂载路径（prim_path）修改为当前环境的命名空间下
    不能用.copy()的原因：
    1.用 copy.deepcopy()（深拷贝），虽然能把嵌套层都复制独立，但 IsaacLab 的配置里包含大量的 USD 路径、物理材质等复杂对象，

    2.replace 的优势：replace 是 dataclasses 原生提供的方法，它专门用于“浅拷贝并覆盖指定字段”。它在复制顶层字段的同时，帮你安全地修改了 prim_path，在 IsaacLab 中是官方推荐的配置派生方式。

    3. 1和2都是deepseek的解释，gpt老师的解释：replace(...) = 基于已有 config 创建一份修改了指定字段的新配置。
    """



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # 创建控制器
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    """
    1. 这个DifferentialIKControllerCfg其实明确需要指定的是这个command_type，用"pose"或者"position",它表示控制完整 pose：位置+姿态；"position" 则只控制 XYZ
    2. 这个use_relative_mode本来就默认是false,其实这里不写也行，当然false表示绝对模式，比如输入的指令是目标在空间中的绝对位姿。控制器会计算出从当前位姿到这个绝对目标位姿所需的关节运动
    true就表示是相对模式,输入的指令是一个增量（Delta），即相对于当前位姿，期望在位置或姿态上变化多少。控制器会计算如何实现这个“变化量”
    3. ik_method这个也是需要明确指定的,有四种解ik方法：ik_method: Literal["pinv", "svd", "trans", "dls"] = MISSING
    这里主要说优点了，详细可以去搜
    pinv (伪逆) -> 理论完备；远离奇异点时收敛速度快，精度高
    svd (奇异值分解) -> 能有效处理奇异位形，避免数值不稳定
    trans (雅可比转置) -> Jacobian transpose 是简单方法，但慢、收敛特性依赖增益，不等于普遍“最稳定”   算得好慢，惊人的慢,浅浅试了一下
    dls (阻尼最小二乘) -> 在奇异点附近非常稳定，能平滑通过奇异区域，运动连续
    """
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device) #这里我先用1了，如果需要改再去加parser.add_argument("--num_envs", type=int, default=128)


    # xyz三个箭头的marker
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))   #再次出现replace

    # 定义机械臂要去的目标，应该isaacsim的定义是[x, y, z, qw, qx, qy, qz]对我的版本来说，但我怎么给出这个ee_goals呢，奥我去那个vscode的URDF预览应该也可以测量一下，就先不打开isaacsim测量了
    # 还有就是这是机器人 base frame 中的目标，不是任意 world frame。 官方就是这么要求的
    ee_goals = [
        [0.4, 0.2, 0.5, 0.707, 0, 0.707, 0],
        [0.5, -0.2, 0.3, 0.707, 0.707, 0.0, 0.0],
        #[0.5, 0.2, 0.3, 0.0, 1.0, 0.0, 0.0], 求不出解？！
        #[0.4, 0.2, 0.3, 0.0, 1.0, 0.0, 0.0], 也求不出来！
        [0.4, 0.3, 0.2, 0.0, 1.0, 0.0, 0.0],  #好奇怪这个也求不出来，因为位姿？
        [0.4, 0.3, 0.2, 0.707, 0, 0.707, 0],  #还真是位姿……
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # 跟踪给定指令，其实我感觉像是在维度统一
    current_goal_idx = 0
    # 创建缓冲区以存储操作
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)#这个action_dim指的是Dimension of the controller's input command. torch.zeros是生成全部为0的坐标
    ik_commands[:] = ee_goals[current_goal_idx]

    # 指定机器人特定参数，但我这里就不写了，我就指定lebai机械臂不放其他机械臂


    # 解析场景实体
    robot_entity_cfg = SceneEntityCfg("robot",joint_names=["joint_[1-6]"],body_names=["gripper_base_link"])
    robot_entity_cfg.resolve(scene)
    # 获得末端执行器的帧索引
    # 对于固定基座机器人，坐标系索引比机身索引少一个。这是因为根部机身未包含在返回的雅可比矩阵中。
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 150 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions  这下面三个是新重置的部分
            ik_commands[:] = ee_goals[current_goal_idx]   
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller  控制器求解也重置
            diff_ik_controller.reset()  #注意这里不是因为控制器保存了某个“IK 解”；主要是清理 controller 内部状态，episode reset 时保持状态一致
            diff_ik_controller.set_command(ik_commands)
            # change goal  去求解下一个点
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            # 从仿真中获取量，获取了jacobian(力雅可比)，ee_pose_w(末端夹具位置姿态),root_pose_w（root_joint姿态），joint_pos(机械臂关节位置)
            # 这里我决定debug看看每个的shape
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]  # ee_jacabi_idx=12
            #我很好奇这个root_physx_view.get_jacobians()和数据结构，让我看看：
            #print(f"看看这个root_physx_view.get_jacobians()的数据结构：{robot.root_physx_view.get_jacobians().shape}")  #torch.Size([1, 13, 6, 12])  这个13高低有点东西，全部轴？应该是的，详细的解释是大致是 env × body-frame × spatial-dim × dof；最后的 12 才是 articulation DOF 维度
            #print(f"看看这个jacobian的结构: {jacobian.shape} ")  # torch.Size([1, 6, 6])  感觉像夹抓的位置，夹爪那6个joint

            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]  #[x, y, z, qw, qx, qy, qz]  位置是gripper_r_link_finger的位置
            root_pose_w = robot.data.root_pose_w  #[x, y, z, qw, qx, qy, qz]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]  #joint[1-6]的位置
            # 计算根坐标系中的坐标系
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )   # ee_pos_b是[x, y, z] ，ee_quat_b是[qw, qx, qy, qz]
            # 计算关节转动多少的命令
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)##第一个count算出来还是原点 [1,6]

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)  
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # 从仿真中获取量,这里获取ee_pose的姿态，给那个箭头显示用
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    #scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene_cfg = TableTopSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
