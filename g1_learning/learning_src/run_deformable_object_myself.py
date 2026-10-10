"""
这是示例脚本，用于展示如何在isaaclab中创建可变形物体且循环操作，
涉及一些变形物体的柏松比和杨氏模量等参数的设置，以及如何在仿真中进行物理交互和控制。
原文件：/home/l/isaac_space_01/IsaacLab/scripts/tutorials/01_assets/run_deformable_object.py
"""

"""
这里需要理解的核心的地方是：

"""
import argparse  #导入 argparse 模块，用于解析命令行参数
from isaaclab.app import AppLauncher #把Isaac Lab应用程序启动器导入到当前脚本中

# 创建参数解析器
parser = argparse.ArgumentParser(description="理解可变形物体仿真的教程。")
# 添加AppLauncher的命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析命令行参数
args_cli = parser.parse_args()
# 启动Omniverse应用程序
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余的代码如下。"""
import torch  # 导入PyTorch库，用于张量计算和深度学习
import isaaclab.sim as sim_utils  # 导入Isaac Lab的仿真
import isaaclab.sim.utils.prims as prim_utils  # 导入Isaac Lab的初始工具
import isaaclab.utils.math as math_utils  # 导入Isaac Lab的数学工具
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim import SimulationContext

def design_scene():
    """设计场景"""
    #生成地面
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # 生成灯光
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # 创建单独的组，称为“Origin1”、“Origin2”、“Origin3”
    # 
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # 定义可变形物体的配置
    cfg = DeformableObjectCfg(
        prim_path="/World/Origin.*/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),  # 红色立方体
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.001,
                # youngs_modulus=10000.0,  # 杨氏模量
                # poissons_ratio=0.4,  # 泊松比  #这两个属于材料配置，不在这个函数
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4,youngs_modulus=100000)
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        debug_vis=True,  #这是一个布尔值，表示是否启用调试可视化。启用调试可视化后，可以在仿真中看到可变形物体的网格、碰撞体和其他调试信息，这对于调试和分析仿真行为非常有帮助
    )
    cube_object = DeformableObject(cfg=cfg)

    # 返回场景信息
    scene_entities = {"cube_object": cube_object}
    return scene_entities, origins

def run_simulation(sim: sim_utils.SimulationContext, entities: dict[str, DeformableObject], origins: torch.Tensor):
    """运行仿真循环"""
    # 获取场景实体
    cube_object = entities["cube_object"]
    # 定义仿真循环的参数
    sim_dt = sim.get_physics_dt()  # 获取仿真时间步长
    sim_time = 0.0  # 初始化仿真时间
    count = 0

    # 可变形体的节点运动目标
    nodal_kinematic_target = cube_object.data.nodal_kinematic_target.clone() # 克隆可变形体的节点运动目标

    # 物理模拟
    while simulation_app.is_running():
        # 重置
        if count % 350 == 0:
            # 重置计数器
            sim_time = 0.0
            count = 0

            # 重置可变形体的节点状态
            nodal_state = cube_object.data.default_nodal_state_w.clone()  # 克隆可变形体的默认节点状态

            
            # 使用 (pos_w, quat_w) 对所有节点的位置执行刚体空间变换
            pos_w = torch.rand(cube_object.num_instances, 3, device=sim.device) * 0.1 + origins  # 在原点附近生成随机位置
            quat_w = math_utils.random_orientation(cube_object.num_instances, device=sim.device)  # 生成随机方向
            nodal_state[..., :3] = cube_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)  # 将随机位置和方向应用到节点状态

            # 将节点状态写入仿真
            cube_object.write_nodal_state_to_sim(nodal_state)  # 将节点状态写入仿真器中

            # 将节点状态写入运动目标，并释放所有顶点
            nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            nodal_kinematic_target[..., 3] = 1.0  # 把所有节点释放，让它们自由参与物理仿真。

            """
            flag = 0
            → kinematically driven
            → 被运动学目标约束


            flag = 1    
            → free
            → 自由节点
            """
            # 所以1.0 表示 free：释放所有节点，不施加运动学约束，不要叫成权重

            cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

            # 重置缓冲区
            cube_object.reset()  # 重置 DeformableObject 内部缓冲区/缓存

            print("----------------------------")
            print("[INFO]: 重置物体状态...")

        # 更新索引0和3处立方体的运动目标
        # 通过选择索引为0的顶点，我们在z方向上稍微移动立方体
        nodal_kinematic_target[[0, 3], 0, 2] += 0.001 
        #这里现在官方案例一般用的，意思明确：位置增量 = 速度 × 时间，，，这个dt的概念非常重要
        # speed = 0.1
        # nodal_kinematic_target[[0, 3], 0, 2] += speed * sim_dt
        
        
        # 将索引为0和3的顶点在z方向上移动0.001个单位
        # 0: 约束，1:自由
        nodal_kinematic_target[[0, 3], 0, 3] = 0.0  # 将索引为0和3的顶点设置为运动约束状态
        # 将运动目标写入仿真
        cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

        # 向模拟写入内部数据
        cube_object.write_data_to_sim()  # 将 Asset 中待写入仿真器的数据/命令同步到仿真器
        # 执行步骤
        sim.step()

        # 通知 DeformableObject：仿真已前进 sim_dt，
        # 更新内部数据时间戳/缓冲区   write → step → update是应该牢牢记住的循环结构
        cube_object.update(sim_dt)

        # 更新仿真时间
        sim_time +=sim_dt
        count += 1
        #
        # 打印初始位置
        if count % 50 == 0:
            #print(f"仿真的初始位置 ： {cube_object.data.root_pos_w[:, :3]}") #这里不要混淆刚体那个13维的状态，可变形体root_pos_w 本身 shape 就是3维，不要那个3都行
            print(f"当前可变形体的平均节点位置 ： {cube_object.data.root_pos_w}")
def main():
    """主函数"""
    # 加载仿真
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # 设置相机
    sim.set_camera_view(eye=[3.0, 0.0, 1.0],target=[0.0,0.0,0.5])
    # 设计场景
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins,device=sim.device)
    # 执行仿真
    sim.reset()
    # 准备好了
    print("[INFO]: Setup complete")
    # 开始仿真
    run_simulation(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # 
    main()
    # ctrl+C关闭
    simulation_app.close()





