"""
2.1 启动一个空的仿真器需要的代码
启动方法是 python g1_learning/learning_src/create_empty_myself.py
"""

"""
这里需要注意的是：

1. AppLauncher 是启动仿真应用的唯一标准方式
    在 Isaac Lab 中编写独立 Python 脚本时，第一步永远是启动仿真应用。
    这是因为 Isaac Sim 的大量依赖模块只有在仿真应用运行之后才可导入。
    而 AppLauncher 就是为此设计的工具类，其设计目的就是为了让用户能通过一个可移植的接口来启动仿真，
    而不必直接处理底层 SimulationApp 的复杂配置

"""

import argparse  # 导入 argparse 模块，用于解析命令行参数
from isaaclab.app import AppLauncher  # 把Isaac Lab应用程序启动器导入到当前脚本中

# 创建参数解析器
parser = argparse.ArgumentParser(description="启动一个空的仿真器")
# 添加AppLauncher的命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析命令行参数
args_cli = parser.parse_args()
# 启动Omniverse应用程序
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余的代码如下。"""
from isaaclab.sim import SimulationCfg, SimulationContext  #从isaaclab.sim模块中导入SimulationCfg和SimulationContext类，用于配置和管理仿真环境

def main():
    """主函数"""

    # 初始化仿真上下文
    sim_cfg = SimulationCfg(dt=0.01)  # 设置仿真配置，dt为时间步长
    sim = SimulationContext(sim_cfg)
    # 设置主摄像机视角  这里第一个参数是摄像机位置，第二个参数是摄像机看向的目标点
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # 播放仿真器
    sim.reset()
    print("[INFO]: Setup complete...")

    # 模拟物理过程
    while simulation_app.is_running():
        sim.step()  # 执行仿真步骤

if __name__ == "__main__":  #如果当前脚本是主程序，则执行以下代码
    main()  # 运行主函数


    simulation_app.close()  # 关闭仿真应用程序