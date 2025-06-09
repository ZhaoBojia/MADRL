# debug field 调试字段
import os  # 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 抑制关于加载多个Intel MKL库副本的警告  （防止在运行多个库时存在版本冲突等导致报错）

import argparse  # 解析命令行参数 （用户通过命令行向脚本传递参数）

# get argument from user 从用户获取参数
parser = argparse.ArgumentParser()  # 创建参数解析器
# 添加参数选项 （在终端输入或运行配置）
parser.add_argument('--drl', type=str, required=True, default='td3', help="which drl algo would you like to choose ['ddpg', 'td3']")  # 用于选择DRL（深度强化学习）算法的参数，默认为'td3'，是一个必需的参数
parser.add_argument('--reward', type=str, required=True, default='see',help="which reward would you like to implement ['ssr', 'see']")  # 用于选择奖励设计的参数，默认为'see'，是一个必需的参数
parser.add_argument('--seeds', type=int, required=False, default=None, nargs='+',help="what seed(s) would you like to use for DRL 1 and 2, please provide in one or two int")  # 用于指定DRL训练的种子参数，可选，默认为None，可以提供一个或两个整数作为参数
parser.add_argument('--ep-num', type=int, required=False, default=300,help="how many episodes do you want to train your DRL")  # 用于指定训练DRL模型的回合数，默认为300
parser.add_argument('--trained-uav', default=False, action='store_true', help='use trained uav instead of retraining')  # 一个标志参数，如果存在，则表示应使用预训练的无人机模型而不是重新训练
# 调用方法，解析传入参数并存储
args = parser.parse_args()
# 将解析的命令行参数赋值给对应变量
DRL_ALGO = args.drl
REWARD_DESIGN = args.reward
SEEDS = args.seeds
EPISODE_NUM = args.ep_num
TRAINED_UAV = args.trained_uav

# process the argument 处理参数 首先它assert语句验证用户选择的是否符合否则报错
assert DRL_ALGO in ['ddpg', 'td3'], "drl must be ['ddpg', 'td3']"
assert REWARD_DESIGN in ['ssr', 'see'], "reward must be ['ssr', 'see']"
if SEEDS is not None:
    assert len(SEEDS) in [1, 2] and isinstance(SEEDS[0], int) and isinstance(SEEDS[-1],int), "seeds must be a list of 1 or 2 integer"  # SEEDS 的长度必须为 1 或 2。SEEDS 的第一个元素和最后一个元素必须是整数类型。如果断言失败，将抛出异常并显示指定的错误消息，以提示用户提供的种子值必须满足特定的条件
# 根据用户选择的DRL_ALGO，从相应的模块中导入Agent类。如果选择的是'td3'算法，则从td3模块中导入Agent类；如果选择的是'ddpg'算法，则从ddpg模块中导入Agent类。
if DRL_ALGO == 'td3':
    from td3 import Agent
elif DRL_ALGO == 'ddpg':
    from ddpg import Agent


import ddpg  # 导入了一个名为 ddpg 的模块。这可能是一个自定义的模块，其中包含了实现 DDPG（Deep Deterministic Policy Gradient）算法所需的代码。

from env_3 import MiniSystem  # 这可能是一个自定义的环境类，用于模拟某种系统或环境。
import numpy as np  # 导入了 NumPy 库，并将其命名为 np，这是 Python 中用于数值计算的常用库。
import math  # 标准数学库，用于执行数学运算。
import time  # 导入了 Python 的时间模块，用于处理时间相关的操作
import torch  # 导入了 PyTorch 库，这是一个用于深度学习的开源机器学习库。

# 1 init system model 初始化系统模型
episode_num = EPISODE_NUM  # recommend to be 300
episode_cnt = 0
step_num = 100

project_name = f'trained_uav/{DRL_ALGO}_{REWARD_DESIGN}' if TRAINED_UAV else f'scratch/{DRL_ALGO}_{REWARD_DESIGN}'  # 是否使用预训练模型决定项目（存储路径）名称
# 创建了一个名为 system 的 MiniSystem 对象，这可能是一个自定义的系统模型类。 初始化仿真环境，多个参数设计
system = MiniSystem(
    user_num=2,
    RIS_ant_num=4,  # 天线数量
    UAV_ant_num=4,
    if_dir_link=1,  # 是否使用定向链接
    if_with_RIS=True,  # 是否包含IRS
    if_move_users=True,  # 是否移动用户
    if_movements=True,  # 是否移动天线
    reverse_x_y=(False, False),
    if_UAV_pos_state=True,  # 是否使用UAV 位置状态
    reward_design=REWARD_DESIGN,  # 奖励设计
    project_name=project_name,  # 项目名称
    step_num=step_num  # 步数
)
# 设置变量 （波束成形角度 增益 基站 是可调节的）
if_Theta_fixed = False
if_G_fixed = False
if_BS = False
if_robust = True  # 可用于鲁棒性训练
# if_power_consumed= False

# 2 init RL Agent 初始化强化学习代理
agent_1_param_dic = {}
agent_1_param_dic["alpha"] = 0.0001  # 学习率 神经网络更新步长
agent_1_param_dic["beta"] = 0.001  # 学习率 控制优化速度/正则化
agent_1_param_dic["input_dims"] = system.get_system_state_dim()  # 输入维度
agent_1_param_dic["tau"] = 0.001  # 软更新参数
agent_1_param_dic["batch_size"] = 64  # 批处理大小
agent_1_param_dic["n_actions"] = system.get_system_action_dim() - 2  # 动作维度
agent_1_param_dic["action_noise_factor"] = 0.1# 动作噪声因子
agent_1_param_dic["memory_max_size"] = int(5 / 5 * episode_num * step_num)  # /2 记忆最大大小
agent_1_param_dic["agent_name"] = "G_and_Phi"  # 代理名称
agent_1_param_dic["layer1_size"] = 800  # 神经网络层大小
agent_1_param_dic["layer2_size"] = 600
agent_1_param_dic["layer3_size"] = 512
agent_1_param_dic["layer4_size"] = 256

agent_2_param_dic = {}
agent_2_param_dic["alpha"] = 0.0001
agent_2_param_dic["beta"] = 0.001
agent_2_param_dic["input_dims"] = 3
agent_2_param_dic["tau"] = 0.001
agent_2_param_dic["batch_size"] = 64
agent_2_param_dic["n_actions"] = 2
agent_2_param_dic["action_noise_factor"] = 0.5
agent_2_param_dic["memory_max_size"] = int(5 / 5 * episode_num * step_num)  # /2
agent_2_param_dic["agent_name"] = "UAV"
agent_2_param_dic["layer1_size"] = 400
agent_2_param_dic["layer2_size"] = 300
agent_2_param_dic["layer3_size"] = 256
agent_2_param_dic["layer4_size"] = 128

agent_3_param_dic = {}
agent_3_param_dic["alpha"] = 0.0001
agent_3_param_dic["beta"] = 0.001
agent_3_param_dic["input_dims"] = system.get_system_state_dim()
agent_3_param_dic["tau"] = 0.001
agent_3_param_dic["batch_size"] = 64
agent_3_param_dic["n_actions"] = system.get_system_action_dim()-2
agent_3_param_dic["action_noise_factor"] = 0.3
agent_3_param_dic["memory_max_size"] = int(5 / 5 * episode_num * step_num)  # /2
agent_3_param_dic["agent_name"] = "Energy"
agent_3_param_dic["layer1_size"] = 400
agent_3_param_dic["layer2_size"] = 300
agent_3_param_dic["layer3_size"] = 256
agent_3_param_dic["layer4_size"] = 128

if SEEDS is not None:  # 设置随机数种子以确保实验的可重复性。两行代码用于设置 PyTorch 的随机数种子，以确保在使用 GPU 时也能实现实验的可重复性。
    torch.manual_seed(SEEDS[0])  # 1 设置了 PyTorch 的 CPU 随机数种子，使得通过 CPU 运行的随机操作在相同的种子下会产生相同的结果。
    torch.cuda.manual_seed_all(SEEDS[0])  # 1
agent_1 = Agent(  # 传入了一系列参数
    alpha=agent_1_param_dic["alpha"],
    beta=agent_1_param_dic["beta"],
    input_dims=[agent_1_param_dic["input_dims"]],
    tau=agent_1_param_dic["tau"],
    env=system,
    batch_size=agent_1_param_dic["batch_size"],
    layer1_size=agent_1_param_dic["layer1_size"],
    layer2_size=agent_1_param_dic["layer2_size"],
    layer3_size=agent_1_param_dic["layer3_size"],
    layer4_size=agent_1_param_dic["layer4_size"],
    n_actions=agent_1_param_dic["n_actions"],
    max_size=agent_1_param_dic["memory_max_size"],
    agent_name=agent_1_param_dic["agent_name"]
)

if SEEDS is not None:
    torch.manual_seed(SEEDS[-1])  # 2
    torch.cuda.manual_seed_all(SEEDS[-1])  # 2
agent_2 = Agent(
    alpha=agent_2_param_dic["alpha"],
    beta=agent_2_param_dic["beta"],
    input_dims=[agent_2_param_dic["input_dims"]],
    tau=agent_2_param_dic["tau"],
    env=system,
    batch_size=agent_2_param_dic["batch_size"],
    layer1_size=agent_2_param_dic["layer1_size"],
    layer2_size=agent_2_param_dic["layer2_size"],
    layer3_size=agent_2_param_dic["layer3_size"],
    layer4_size=agent_2_param_dic["layer4_size"],
    n_actions=agent_2_param_dic["n_actions"],
    max_size=agent_2_param_dic["memory_max_size"],
    agent_name=agent_2_param_dic["agent_name"]
)

if SEEDS is not None:  # 设置随机数种子以确保实验的可重复性。两行代码用于设置 PyTorch 的随机数种子，以确保在使用 GPU 时也能实现实验的可重复性。
    torch.manual_seed(SEEDS[1])  # 3 设置了 PyTorch 的 CPU 随机数种子，使得通过 CPU 运行的随机操作在相同的种子下会产生相同的结果。
    torch.cuda.manual_seed_all(SEEDS[1])  # 3
agent_3 = Agent(  # 传入了一系列参数
    alpha=agent_3_param_dic["alpha"],
    beta=agent_3_param_dic["beta"],
    input_dims=[agent_3_param_dic["input_dims"]],
    tau=agent_3_param_dic["tau"],
    env=system,
    batch_size=agent_3_param_dic["batch_size"],
    layer1_size=agent_3_param_dic["layer1_size"],
    layer2_size=agent_3_param_dic["layer2_size"],
    layer3_size=agent_3_param_dic["layer3_size"],
    layer4_size=agent_3_param_dic["layer4_size"],
    n_actions=agent_3_param_dic["n_actions"],
    max_size=agent_3_param_dic["memory_max_size"],
    agent_name=agent_3_param_dic["agent_name"]
)
# 这段代码根据是否使用经过训练的 UAV 模型，加载了对应的预训练模型。 加载模型后，这些模型将被用作 UAV 代理 agent_2 的策略网络和值函数网络，以便直接在已训练的基础上进行进一步的学习或测试。
if TRAINED_UAV:
    benchmark = f'data/storage/benchmark/{DRL_ALGO}_{REWARD_DESIGN}_benchmark'
    if DRL_ALGO == 'td3':
        agent_2.load_models(
            load_file_actor=benchmark + '/Actor_UAV_TD3',
            load_file_critic_1=benchmark + '/Critic_1_UAV_TD3',
            load_file_critic_2=benchmark + '/Critic_2_UAV_TD3'
        )
    elif DRL_ALGO == 'ddpg':
        agent_2.load_models(
            load_file_actor=benchmark + '/Actor_UAV_ddpg',
            load_file_critic=benchmark + '/Critic_UAV_ddpg'
        )

meta_dic = {}  # 打印系统信息并将其保存到 meta_dic 字典中。
print("***********************system information******************************")
print("folder_name:     " + str(system.data_manager.store_path))  # 系统数据存储的文件夹名称
meta_dic['folder_name'] = system.data_manager.store_path
print("user_num:        " + str(system.user_num))  # 用户数量
meta_dic['user_num'] = system.user_num
print("if_dir:          " + str(system.if_dir_link))  # 是否使用定向链接
meta_dic['if_dir_link'] = system.if_dir_link
print("if_with_RIS:     " + str(system.if_with_RIS))  # 是否包含RIS信息
meta_dic['if_with_RIS'] = system.if_with_RIS
print("if_user_m:       " + str(system.if_move_users))  # 用户是否移动
meta_dic['if_move_users'] = system.if_move_users
print("RIS_ant_num:     " + str(system.RIS.ant_num))
meta_dic['system_RIS_ant_num'] = system.RIS.ant_num  # RIS天线数量
print("UAV_ant_num:     " + str(system.UAV.ant_num))
meta_dic['system_UAV_ant_num'] = system.UAV.ant_num  # UAV天线数量
print("if_movements:    " + str(system.if_movements))
meta_dic['system_if_movements'] = system.if_movements
print("reverse_x_y:     " + str(system.reverse_x_y))  # 坐标轴是否反转
meta_dic['system_reverse_x_y'] = system.reverse_x_y
print("if_UAV_pos_state:" + str(system.if_UAV_pos_state))  # 是否使用 UAV 位置状态的信息
meta_dic['if_UAV_pos_state'] = system.if_UAV_pos_state
print("ep_num:          " + str(episode_num))  # 回合数
meta_dic['episode_num'] = episode_num
print("step_num:        " + str(step_num))  # 步数
meta_dic['step_num'] = step_num
print("***********************agent_1 information******************************")
tplt = "{0:{2}^20}\t{1:{2}^20}"  # 用了一个格式化字符串 tplt，将参数名和参数值打印在一行中，并使用中文空格 chr(12288) 进行填充，使输出更加对齐美观。
for i in agent_1_param_dic:
    parm = agent_1_param_dic[i]  # 使用 agent_1_param_dic 字典中的参数名作为键来遍历字典，并打印每个参数的名称和对应的值。
    print(tplt.format(i, parm, chr(12288)))
meta_dic["agent_1"] = agent_1_param_dic

print("***********************agent_2 information******************************")
for i in agent_2_param_dic:
    parm = agent_2_param_dic[i]
    print(tplt.format(i, parm, chr(12288)))
meta_dic["agent_2"] = agent_2_param_dic

print("***********************agent_3 information******************************")
for i in agent_3_param_dic:
    parm = agent_3_param_dic[i]
    print(tplt.format(i, parm, chr(12288)))
meta_dic["agent_3"] = agent_3_param_dic
system.data_manager.save_meta_data(
    meta_dic)  # 它将之前构建的 meta_dic 字典作为参数传递给了 save_meta_data 方法。这个方法会将 meta_dic 中的信息保存到文件中，以便在之后的实验过程中能够方便地获取和使用这些信息。

print("***********************traning information******************************")

while episode_cnt < episode_num:
    # 1 reset the whole system 重置系统状态 以便开始新的训练回合
    system.reset()
    step_cnt = 0  # 回合步数
    score_per_ep = 0  # 得分相关的变量

    # 2 get the initial state 初始状态
    if if_robust:  # 稳健性设置 (是否对系统状态进行扰动）
        tmp = system.observe()  # 调用 system.observe() 方法获取当前系统的观察结果，并将其存储在变量 tmp 中。
        # z = np.random.multivariate_normal(np.zeros(2), 0.5*np.eye(2), size=len(tmp)).view(np.complex128)
        z = np.random.normal(size=len(tmp))
        observersion_1 = list(
            np.array(tmp) + 0.6 * 1e-7 * z  # 将对观察结果进行扰动。通过生成服从正态分布的随机数 z，并将其与观察结果相加，以产生带有扰动的观察结果。扰动的幅度由参数 0.6 *1e-7 控制。
        )
    else:
     observersion_1 = system.observe()
    observersion_2 = list(system.UAV.coordinate)  # 无人机坐标
    observersion_3 = system.observe()

    if episode_cnt == 80:
        print("break point")

    while step_cnt < step_num:
        # 1 count num of step in one episode
        step_cnt += 1

        # 判断是否暂停系统
        if not system.render_obj.pause:
            # 2 choose action according to current state
            action_1 = agent_1.choose_action(observersion_1, greedy=agent_1_param_dic["action_noise_factor"] * math.pow(
                (1 - episode_cnt / episode_num), 2))
            action_2 = agent_2.choose_action(observersion_2, greedy=agent_2_param_dic["action_noise_factor"] * math.pow(
                (1 - episode_cnt / episode_num), 2))
            action_3 = agent_3.choose_action(observersion_3, greedy=agent_3_param_dic["action_noise_factor"] * math.pow(
                (1 - episode_cnt / episode_num), 2))

            if if_BS:
                action_2[0] = 0
                action_2[1] = 0

            if if_Theta_fixed:  # 固定角度
                action_1[0 + 2 * system.UAV.ant_num * system.user_num:] = len(
                    action_1[0 + 2 * system.UAV.ant_num * system.user_num:]) * [0]

            if if_G_fixed:  # 固定通信通道增益
                action_1[0:0 + 2 * system.UAV.ant_num * system.user_num] = np.array(
                    [-0.0313, -0.9838, 0.3210, 1.0, -0.9786, -0.1448, 0.3518, 0.5813, -1.0, -0.2803, -0.4616, -0.6352,
                     -0.1449, 0.7040, 0.4090, -0.8521]) * math.pow(episode_cnt / episode_num, 2) * 0.7

            # 3 get newstate, reward
            if system.if_with_RIS:
                new_state_1, reward, done, info = system.step(
                    action_0=action_2[0],  # UAV的X, Y坐标
                    action_1=action_2[1],
                    G=action_1[0:0 + 2 * system.UAV.ant_num * system.user_num],  # 增益
                    Phi=action_1[0 + 2 * system.UAV.ant_num * system.user_num:],  # 动作参数
                    set_pos_x=action_2[0],  # ris相位
                    set_pos_y=action_2[1]
                )
                new_state_2 = list(system.UAV.coordinate)
                new_state_3 = system.observe()
            else:
                new_state_1, reward, done, info = system.step(
                    action_0=action_2[0],
                    action_1=action_2[1],
                    G=action_1[0:0 + 2 * system.UAV.ant_num * system.user_num],
                    set_pos_x=action_2[0],
                    set_pos_y=action_2[1]
                )
                new_state_2 = list(system.UAV.coordinate)
                new_state_3 = system.observe()

            # 4 store state pair into mem pool 存储状态到记忆池
            agent_1.remember(observersion_1, action_1, reward, new_state_1, int(done))
            agent_2.remember(observersion_2, action_2, reward, new_state_2, int(done))
            agent_3.remember(observersion_3, action_3, reward, new_state_3, int(done))

            # 5 update DDPG net 更新深度神经网络
            agent_1.learn()
            if not TRAINED_UAV:
                agent_2.learn()
            agent_3.learn()  # 更新agent_3的神经网络

            # 更新得分
            score_per_ep += reward  # 累加每步的奖励

            # 更新状态
            observersion_1 = new_state_1
            observersion_2 = new_state_2
            observersion_3 = new_state_3  # 假设agent_3的状态更新独立，具体可根据需求

            if done:
                break

        else:
            time.sleep(0.001)

    # === 保存当前 episode 的 Loss 数据（三智能体结构） ===
    # === Agent 1 ===
    if "actor_loss_list_agent_1" not in system.data_manager.simulation_result_dic:
        system.data_manager.simulation_result_dic["actor_loss_list_agent_1"] = []
    system.data_manager.simulation_result_dic["actor_loss_list_agent_1"].append(agent_1.loss_actor_history)

    if "critic_loss_1_list_agent_1" not in system.data_manager.simulation_result_dic:
        system.data_manager.simulation_result_dic["critic_loss_1_list_agent_1"] = []
    system.data_manager.simulation_result_dic["critic_loss_1_list_agent_1"].append(agent_1.loss_critic_1_history)

    if "critic_loss_2_list_agent_1" not in system.data_manager.simulation_result_dic:
        system.data_manager.simulation_result_dic["critic_loss_2_list_agent_1"] = []
    system.data_manager.simulation_result_dic["critic_loss_2_list_agent_1"].append(agent_1.loss_critic_2_history)

    if "critic_loss_list_agent_1" not in system.data_manager.simulation_result_dic:
        system.data_manager.simulation_result_dic["critic_loss_list_agent_1"] = []
    system.data_manager.simulation_result_dic["critic_loss_list_agent_1"].append(agent_1.loss_critic_total_history)

    # === Agent 2 ===
    if "actor_loss_list_agent_2" not in system.data_manager.simulation_result_dic:
        system.data_manager.simulation_result_dic["actor_loss_list_agent_2"] = []
    system.data_manager.simulation_result_dic["actor_loss_list_agent_2"].append(agent_2.loss_actor_history)

    if "critic_loss_1_list_agent_2" not in system.data_manager.simulation_result_dic:
        system.data_manager.simulation_result_dic["critic_loss_1_list_agent_2"] = []
    system.data_manager.simulation_result_dic["critic_loss_1_list_agent_2"].append(agent_2.loss_critic_1_history)

    if "critic_loss_2_list_agent_2" not in system.data_manager.simulation_result_dic:
        system.data_manager.simulation_result_dic["critic_loss_2_list_agent_2"] = []
    system.data_manager.simulation_result_dic["critic_loss_2_list_agent_2"].append(agent_2.loss_critic_2_history)

    if "critic_loss_list_agent_2" not in system.data_manager.simulation_result_dic:
        system.data_manager.simulation_result_dic["critic_loss_list_agent_2"] = []
    system.data_manager.simulation_result_dic["critic_loss_list_agent_2"].append(agent_2.loss_critic_total_history)

    # === Agent 3 ===
    if "actor_loss_list_agent_3" not in system.data_manager.simulation_result_dic:
        system.data_manager.simulation_result_dic["actor_loss_list_agent_3"] = []
    system.data_manager.simulation_result_dic["actor_loss_list_agent_3"].append(agent_3.loss_actor_history)

    if "critic_loss_1_list_agent_3" not in system.data_manager.simulation_result_dic:
        system.data_manager.simulation_result_dic["critic_loss_1_list_agent_3"] = []
    system.data_manager.simulation_result_dic["critic_loss_1_list_agent_3"].append(agent_3.loss_critic_1_history)

    if "critic_loss_2_list_agent_3" not in system.data_manager.simulation_result_dic:
        system.data_manager.simulation_result_dic["critic_loss_2_list_agent_3"] = []
    system.data_manager.simulation_result_dic["critic_loss_2_list_agent_3"].append(agent_3.loss_critic_2_history)

    if "critic_loss_list_agent_3" not in system.data_manager.simulation_result_dic:
        system.data_manager.simulation_result_dic["critic_loss_list_agent_3"] = []
    system.data_manager.simulation_result_dic["critic_loss_list_agent_3"].append(agent_3.loss_critic_total_history)
    print("[Debug] 已写入 loss 字段：",
          [k for k in system.data_manager.simulation_result_dic if "loss" in k])

    # 保存每回合数据
    system.data_manager.save_file(episode_cnt=episode_cnt)
    system.reset()
    print("ep_num: " + str(episode_cnt) + "   ep_score:  " + str(score_per_ep))
    episode_cnt += 1

    # 每10轮保存模型
    if episode_cnt % 10 == 0:
        agent_1.save_models()
        agent_2.save_models()
        agent_3.save_models()  # 保存智能体3的模型

# 保存最后的模型
agent_1.save_models()
agent_2.save_models()
agent_3.save_models()  # 保存智能体3的最终模型

