"""
相关文件：
1. 基本配置文件和导入Manager：/home/l/isaac_space_01/IsaacLab/source/isaaclab_tasks/
isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py
2.  MDP嵌套文件，这里可以找到一个特定的奖励函数：
/home/l/isaac_space_01/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/mdp/rewards.py
3. 这里的文件可以看到很多的预定义的MDP术语：
/home/l/isaac_space_01/IsaacLab/source/isaaclab/isaaclab/envs/mdp
4. 
"""

"""
启动形式 python /home/l/isaac_space_01/g1_learning/learning_src/create_cartpole_MDP_myself.py  
/home/l/isaac_space_01/g1_learning/learning_src/create_cartpole_MDP_myself.py --num_envs=32
"""

import argparse
from isaaclab.app import AppLauncher

# 加入解析参数
parser = argparse.ArgumentParser(description="关于运行cartpole RL环境的教程")
parser.add_argument("--num_envs",type=int ,default=16,help="生成环境数量")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其他代码"""
import torch 
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg

def main():
    #创建环境配置
    env_cfg = CartpoleEnvCfg()  #这个环境用的是cartpole_env_cfg.py的class CartpoleSceneCfg(InteractiveSceneCfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # 建立RL环境
    """
    这里学习一下ManagerBasedENV和ManagerBasedRLEnv:
    1. ManagerBasedEnv 是一个基础环境，它不包含强化学习（RL）所需的任务特定信息，如奖励（reward）和终止条件（termination）
    2. 而 ManagerBasedRLEnv 继承自 ManagerBasedEnv，并在其基础上添加了 RL 任务所需的功能，例如奖励计算、终止判断、课程学习和命令生成等。
    因此，ManagerBasedEnv 适用于传统的机器人控制，而 ManagerBasedRLEnv 则用于强化学习任务。
    更详细一些的解释在MD文件的2.5处
    """
    env = ManagerBasedRLEnv(cfg=env_cfg)  #manager_based_env.py定义了很多

    count = 0
    while simulation_app.is_running():
        # 禁用 PyTorch 自动求导，环境仿真本身不需要构建计算图，
        # 可以减少开销
        with torch.inference_mode():
            #写入状态
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" *80)  #打印80个“-”
                print("[INFO]: Resetting environment")

            # 随机行为样本写入
            joint_efforts = torch.randn_like(env.action_manager.action)  #Shape is (num_envs, total_action_dim)
            # 向环境输入一次 action，并推进一个 RL environment step
            obs,rew,terminated,truncated,info = env.step(joint_efforts)  #后面只打印了obs
            """
            这里根据cartpole_env_cfg.py的设置：
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras
            observation 观测项是joint_pos_rel，joint_vel_rel，并且enable_corruption=False 表示关闭 observation noise/corruption
            reward 奖励有5个过程，第一个是判断进程是否还在，奖励权重是1.0,第二个是惩罚不在运行weight=-2.0,第三个是是否保持直立，weight=-1.0
            第四个是较低推车速度任务，weight=-0.01，第五个是更低的小车角速度weight=-0.005
            terminated 
            → 任务本身触发终止条件
            → Cartpole 例如 cart 越界
            truncated  
            → time_out
            → episode 达到最大时长
            info  我感觉这里是在了另外一个模块定义的，不是在env_cfg这类文件中
            """
            # 打印当前pole的朝向
            print("[Env 0]: Pole joint: ",obs["policy"][0][1].item())   #这里的意思是obs["policy"].shape=(num_envs, obs_dim)，
            """
            index 0 → cart joint position
            index 1 → pole joint position

            index 2 → cart joint velocity
            index 3 → pole joint velocity

            以后别盲猜，直接这样看：
            print(env.scene["robot"].joint_names)
            print(obs["policy"].shape)
            """
            print(env.scene["robot"].joint_names)
            print(obs["policy"].shape)
            # 然后这里的item是因为原本是tensor的形式，.item表示转成普通的python数字
            count += 1

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()

#补充说明：
"""
terminated
=
任务自身进入终止状态

例如：
机器人摔倒
抓取失败
物体掉落
Cartpole 小车跑出边界
任务成功


truncated
=
外部原因把 episode 截断

最典型：
时间到了


对于info:
info
=
不参与主要 observation/action/reward 接口的额外信息
+
日志
+
统计量

"""

"""
这里添加一下MDP的关键部分理解：
结合这个文件看`/home/l/isaac_space_01/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py`
在MDP的定义文件夹里
1. observation.py：
    `/home/l/isaac_space_01/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py`
    我找到了MDP的observation.py的joint_pos_rel和joint_vel_rel的定义，rel的全称是relative,相对的，所以返回的是相减
    这里的定义return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    里的joint_ids的定义是  特定关节或者所有关节，文件在：
    `/home/l/isaac_space_01/IsaacLab/source/isaaclab/isaaclab/managers/scene_entity_cfg.py`

2 rewards.py:
    `/home/l/isaac_space_01/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py`
    我找到了func=mdp.joint_vel_l1的joint_vel_l1的定义，我看见返回的是绝对值的相加，如果是多维可以选维度，如果dim=None，全部维度都相减然后取绝对值再相加
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    但我并没有找到joint_pos_target_l2的定义在这个rewards.py文件里，那这里岂不是其实没意义？其实这是在自己定义的reward里：
    `/home/l/isaac_space_01/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/mdp/rewards.py`
    return torch.sum(torch.square(joint_pos - target), dim=1)
    同时这里认真感受学习一下哪一部分的reward在哪里，哪些是本俩就有的reward，哪些是根据你自己任务的需要的reward

3 terminations.py:
    `/home/l/isaac_space_01/IsaacLab/source/isaaclab/isaaclab/envs/mdp/terminations.py`
    我找到了mdp.time_out的time_out的定义，返回的是应该是布尔值，没超出设定的最大的episode则返回1（当然是tensor的格式）
    return env.episode_length_buf >= env.max_episode_length
    还有mdp.joint_pos_out_of_manual_limit的定义，关节位置是否超过最大最小值，这里用的or:
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)

4 events.py:
    `/home/l/isaac_space_01/IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py`
    我找到了mdp.reset_joints_by_offset的reset_joints_by_offset的定义，在cartpole_env_cfg的reset定义主要用下面这
    三个参数定义了slider_to_cart和cart_to_pole的随机范围：
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")

5 mdp/actions/actions_cfg.py:
    `/home/l/isaac_space_01/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py`
    我找到了关于scale的定义：“动作的比例因子（regex表达式中的float或dict）。默认值为1.0。”
"""