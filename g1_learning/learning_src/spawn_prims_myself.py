# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher #把Isaac Lab应用程序启动器导入到当前脚本中

# 创建参数解析器
parser = argparse.ArgumentParser(description="理解如何在场景中生成prim的教程。")
# 添加AppLauncher的命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析命令行参数
args_cli = parser.parse_args()
# 启动Omniverse应用程序
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余的代码如下。"""
import isaaclab.sim as sim_utils  # 导入Isaac Lab的仿真工具
import isaaclab.sim.utils.prims as prim_utils  # 导入Isaac Lab的初始工具
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # 导入Isaac Lab的资产工具
# import torch  # 导入PyTorch库，用于张量计算和深度学习
# import isaaclab.utils.math as math_utils  # 导入Isaac Lab的数学工具，用于位置和方向的转换计算
# from isaaclab.assets import RigidObject, RigidObjectCfg

def design_scene():
    """设计场景，通过生成地面平面、光源、对象和USD文件中的网格。"""
    # 生成地面平面
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # 生成远程光源
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # 创建一个新的Xform prim，用于生成所有对象
    prim_utils.create_prim("/World/Objects", "Xform")
    # 生成一个红色的圆锥
    cfg_cone = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),#红色圆锥
    )
    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))
    cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))

    # 生成一个带有碰撞体和刚体的绿色圆锥体
    cfg_cone_rigid = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(
            # collision_enabled=True,
            # collision_group=1,
            # collision_mask=1,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )

    cfg_cone_rigid.func(
        "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(-0.2, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
    )   

    # 生成一个可变形体的蓝色立方体
    cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.5, 0.2),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cuboid_deformable.func("/World/Objects/CuboidDeformable", cfg_cuboid_deformable, translation=(0.15, 0.0, 2.0))

    #cfg_cuboid_deformable.reset()  # 重置可变形体的状态，以便在每个仿真步骤中保持其初始状态

    # 将表格的 USD 文件生成到场景中
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))

    # #返回场景实体和原点位置
    # scene_entities = {
    #     "Cone1": cfg_cone,
    #     "Cone2": cfg_cone,
    #     "ConeRigid": cfg_cone_rigid,
    #     "CuboidDeformable": cfg_cuboid_deformable,
    # }
    # return scene_entities,cfg_cone_rigid.func

# #尝试写一个函数来运行仿真，循环掉落
# def run_simulation(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
#     """运行仿真循环。"""
#     # 提取场景实体
#     # 注意：我们在这里这样做只是为了可读性。通常，最好直接从字典中访问实体。
#     # 这个字典在下一个教程中被 InteractiveScene 类所取代。
#     cfg_cuboid_deformable = entities["CuboidDeformable"]
#     # 定义仿真步长
#     sim_dt = sim.get_physics_dt()
#     sim_time = 0.0
#     count = 0
#     # 模拟物理
#     while simulation_app.is_running():
#         # 重置
#         if count % 250 == 0:
#             # 重置计数器
#             sim_time = 0.0
#             count = 0
#             # 重置根状态
#             root_state = cfg_cuboid_deformable.data.default_root_state.clone()
#             # 随意取一个圆柱体上的位置
#             root_state[:, :3] += origins
#             root_state[:, :3] += math_utils.sample_cylinder(
#                 radius=1.0, h_range=(0.25, 0.5), size=cfg_cuboid_deformable.num_instances, device=cfg_cuboid_deformable.device
#             )
#             # 将根状态写入仿真
#             cfg_cuboid_deformable.write_root_pose_to_sim(root_state[:, :7])
#             cfg_cuboid_deformable.write_root_velocity_to_sim(root_state[:, 7:])
#             # 重置缓冲区
#             cfg_cuboid_deformable.reset()   #原来重置在这里，重置了位置和速度等状态
#             print("----------------------------------------")
#             print("[INFO]: Resetting object state...")
#         # 应用仿真数据
#         cfg_cuboid_deformable.write_data_to_sim()   #写入数据到仿真器中，更新位置和速度等状态，一些力什么的
#         # 执行步骤
#         sim.step()
#         # 更新仿真时间
#         sim_time += sim_dt
#         count += 1
#         # 更新缓冲区
#         cfg_cuboid_deformable.update(sim_dt)    
#         # 打印根位置
#         print(f"[INFO]: Root position: {cfg_cuboid_deformable.data.root_poses[:, :3]}")


def main():
    """主函数。"""

    # 初始化仿真上下文
    sim_cfg = sim_utils.SimulationCfg(dt=0.01,device=args_cli.device)  # 设置仿真配置，dt为时间步长
    sim = sim_utils.SimulationContext(sim_cfg)
    # 设置主摄像机
    sim.set_camera_view([2.5, 0.0, 2.5], [0.0, 0.0, 0.5])

    # 设计场景
   
    design_scene()

    # scene_entities, scene_origins = design_scene()
    # scene_origins = torch.tensor(scene_origins, device=sim.device)


    # 播放仿真器
    sim.reset()
    # 现在我们准备好了！
    print("[INFO]: Setup complete...")

    # 模拟物理
    while simulation_app.is_running():
        # 执行步骤
        sim.step()
    


        


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用程序
    simulation_app.close()

