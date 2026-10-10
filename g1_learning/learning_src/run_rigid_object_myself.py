"""
这是模仿 IsaacLab/scripts/tutorials/01_assets/run_rigid_object.py 的一个版本，主要是为了学习和理解如何在Isaac Lab中运行刚体对象的仿真。
启动方式就是python g1_learning/run_rigid_object_myself.py
"""

"""
这里需要理解的核心的地方是：

"""


import argparse  #导入 argparse 模块，用于解析命令行参数

from isaaclab.app import AppLauncher #把Isaac Lab应用程序启动器导入到当前脚本中

# 创建参数解析器
parser = argparse.ArgumentParser(description="理解刚体运动仿真的教程。")
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
import torch  # 导入PyTorch库，用于张量计算和深度学习
import isaaclab.utils.math as math_utils  # 导入Isaac Lab的数学工具，用于位置和方向的转换计算
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

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

    origin = [[0.0, 0.0, 2.0]]
    prim_utils.create_prim("/World/Objects", "Xform", translation=origin)

    # 生成一个带有碰撞体和刚体的蓝色长方体
    cube_cfg = RigidObjectCfg(
        prim_path="/World/Objects/CubeRigid",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),  # 蓝色长方体
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )
    cube = RigidObject(cfg=cube_cfg)

    #创建asset,前面都在写物体配置，后面是创建asset对象，asset对象是对物体的封装，里面有很多方法可以操作物体，比如写入位置，速度等
    # 返回场景实体和原点位置
    scene_entities = {
        "CubeRigid": cube,
    }
    return scene_entities, origin


def run_simulation(sim: sim_utils.SimulationContext, entities: dict[str,RigidObject], origins: torch.Tensor):
    """运行仿真器，循环掉落物体。"""
    cube = entities["CubeRigid"]
    #定义仿真步长
    sim_dt = sim.get_physics_dt()  #这是获取仿真器的物理时间步长，单位是秒
    sim_time = 0.0
    count = 0

    # 模拟物理
    while simulation_app.is_running():  
        if count % 250 == 0:
            root_state = cube.data.default_root_state.clone()  #这是获取物体的默认根状态，包含位置、方向、线速度和角速度等信息
            cube.write_root_pose_to_sim(
                root_state[:, :7]   
            )   #这是把物体的根状态写入仿真器中，更新物体的位置和方向等信息
            cube.write_root_velocity_to_sim(
                root_state[:, 7:]
            )  #这是把物体的根速度写入仿真器中，更新物体的线速度和角速度等信息
            cube.reset()  #这是重置物体的状态，清除物体的力和碰撞等信息
            # print("----------------------------------------")
            # print("[INFO]: Resetting object state...")

        sim.step()

        cube.update(sim.get_physics_dt())

        count += 1

def main():
    """主函数。"""

    # 初始化仿真上下文
    sim_cfg = sim_utils.SimulationCfg(dt=0.01,device=args_cli.device)  # 设置仿真配置，dt为时间步长
    sim = sim_utils.SimulationContext(sim_cfg)
    # 设置主摄像机
    sim.set_camera_view([2.5, 0.0, 2.5], [0.0, 0.0, 0.5])

    # 设计场景

    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)


    # 播放仿真器
    sim.reset()
    # 现在我们准备好了！
    print("[INFO]: Setup complete...")

    # 运行仿真器
    run_simulation(sim, scene_entities, scene_origins)


        


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用程序
    simulation_app.close()

