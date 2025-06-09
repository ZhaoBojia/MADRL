# %matplotlib inline                            系统模型 问题公式                        小型ris通信系统
from channel import *
from data_manager import DataManager
from entity import *
from math_tool import *
from render import Render

# s.t every simulition is the same model
np.random.seed(2)

######################################################
# new for energy
# energy related parameters of rotary-wing UAV
# based on Energy Minimization in Internet-of-Things System Based on Rotary-Wing UAV
P_i = 790.6715
P_0 = 580.65
U2_tip = (200) ** 2
s = 0.05
d_0 = 0.3
p = 1.225
A = 0.79
delta_time = 0.1 / 1000  # 0.1ms

# add ons hover veloctiy (添加悬停速度的计算)
# based on https://www.intechopen.com/chapters/57483
m = 1.3  # mass: assume 1.3kg https://www.droneblog.com/average-weights-of-common-types-of-drones/#:~:text=In%20most%20cases%2C%20toy%20drones,What%20is%20this%3F
g = 9.81  # gravity
T = m * g  # thrust 推力
v_0 = (T / (A * 2 * p)) ** 0.5  # （悬停时的垂直速度）


def get_energy_consumption(v_t):  # 计算 UAV 在每个时间槽中的能源消耗  评估 UAV 的能源效率或优化飞行路径。
    '''
    arg
    1) v_t = displacement per time slot #（时间槽位移）
    '''
    energy_1 = P_0 \
               + 3 * P_0 * (abs(v_t)) ** 2 / U2_tip \
               + 0.5 * d_0 * p * s * A * (abs(v_t)) ** 3

    energy_2 = P_i * ((
                              (1 + (abs(v_t) ** 4) / (4 * (v_0 ** 4))) ** 0.5
                              - (abs(v_t) ** 2) / (2 * (v_0 ** 2))
                          ) ** 0.5)

    energy = delta_time * (energy_1 + energy_2)
    return energy


######################################################


class MiniSystem(object):  # 小型RIS通信系统
    # class MiniSystem(K=1):
    """
    define mini RIS communication system with one UAV  # fre = 28e9` 表示通信系统的工作频率为 28 GHz。在给定的代码中，`if_dir_link = 1` 可能表示是否存在直射链路。这个参数的值为1时，可能表示存在直射链路；值为0时，可能表示不存在直射链路。
        and one RIS and one user, one attacker
    """

    def __init__(self, UAV_num=1, RIS_num=1, user_num=1, attacker_num=1, fre=28e9,
                 RIS_ant_num=16, UAV_ant_num=8, if_dir_link=1, if_with_RIS=True,
                 if_move_users=True, if_movements=True, reverse_x_y=(True, True),
                 if_UAV_pos_state=True, if_local_obs=False, reward_design='ssr', project_name=None, step_num=100):
        self.if_dir_link = if_dir_link
        self.if_with_RIS = if_with_RIS
        self.if_move_users = if_move_users
        self.if_movements = if_movements
        self.if_UAV_pos_state = if_UAV_pos_state
        self.if_local_obs = if_local_obs
        self.reverse_x_y = reverse_x_y
        self.user_num = user_num
        self.attacker_num = attacker_num
        self.border = [(-25, 25), (0, 50)]  # 系统边界
        # 1.init entities: 1 UAV, 1 RIS, many users and attackers
        self.data_manager = DataManager(file_path='./data', project_name=project_name,
                                        store_list=['beamforming_matrix', 'reflecting_coefficient', 'UAV_state',
                                                    'user_capacity', 'secure_capacity', 'attaker_capacity', 'G_power',
                                                    'reward', 'UAV_movement'])
        # 1.1 init UAV position and beamforming matrix
        self.UAV = UAV(
            coordinate=self.data_manager.read_init_location('UAV', 0),
            ant_num=UAV_ant_num,
            max_movement_per_time_slot=0.25)  # （每个间隙无人机可移动最大距离）
        self.UAV.G = np.mat(np.ones((self.UAV.ant_num, user_num), dtype=complex), dtype=complex)  # UAV与多用户间的信道增益
        self.power_factor = 100
        self.UAV.G_Pmax = np.trace(self.UAV.G * self.UAV.G.H) * self.power_factor
        # 1.2 init RIS (coor_sys_z：RIS法向量，用于定义反射特性)
        self.RIS = RIS(
            coordinate=self.data_manager.read_init_location('RIS', 0),
            coor_sys_z=self.data_manager.read_init_location('RIS_norm_vec', 0),
            ant_num=RIS_ant_num)
        # 1.3 init users
        self.user_list = []

        for i in range(user_num):
            user_coordinate = self.data_manager.read_init_location('user', i)
            user = User(coordinate=user_coordinate, index=i)
            user.noise_power = -114
            self.user_list.append(user)

        # 1.4 init attackers
        self.attacker_list = []

        for i in range(attacker_num):
            attacker_coordinate = self.data_manager.read_init_location('attacker', i)
            attacker = Attacker(coordinate=attacker_coordinate, index=i)
            attacker.capacity = np.zeros((user_num))  # （攻击者容量 模拟攻击者行为）
            attacker.noise_power = -114
            self.attacker_list.append(attacker)
        # 1.5 generate the eavesdrop capacity array , shape: P X K    生成了一个形状为 P x K 的数组，其中 P 表示攻击者数量，K 表示用户数量。
        self.eavesdrop_capacity_array = np.zeros((attacker_num, user_num))

        # 1.6 reward design
        self.reward_design = reward_design  # reward_design is ['ssr' or 'see']

        # 1.7 step_num
        self.step_num = step_num

        # 2.init channel
        self.H_UR = mmWave_channel(self.UAV, self.RIS, fre)
        self.h_U_k = []
        self.h_R_k = []
        self.h_U_p = []
        self.h_R_p = []
        for user_k in self.user_list:
            self.h_U_k.append(mmWave_channel(user_k, self.UAV, fre))
            self.h_R_k.append(mmWave_channel(user_k, self.RIS, fre))
        for attacker_p in self.attacker_list:
            self.h_U_p.append(mmWave_channel(attacker_p, self.UAV, fre))
            self.h_R_p.append(mmWave_channel(attacker_p, self.RIS, fre))

        # 3 update user and attaker channel capacity
        self.update_channel_capacity()

        # 4 draw system
        self.render_obj = Render(self)

    def reset(self):  # (重置通信系统组件，，将实体恢复初始状态，以便开始新的仿真)
        """
        reset UAV, users, attackers, beamforming matrix, reflecting coefficient
        """
        # 1 reset UAV
        self.UAV.reset(coordinate=self.data_manager.read_init_location('UAV', 0))
        # 2 reset users
        for i in range(self.user_num):
            user_coordinate = self.data_manager.read_init_location('user', i)
            self.user_list[i].reset(coordinate=user_coordinate)
        # 3 reset attackers
        for i in range(self.attacker_num):
            attacker_coordinate = self.data_manager.read_init_location('attacker', i)
            self.attacker_list[i].reset(coordinate=attacker_coordinate)
        # 4 reset beamforming matrix
        self.UAV.G = np.mat(np.ones((self.UAV.ant_num, self.user_num), dtype=complex), dtype=complex)
        self.UAV.G_Pmax = np.trace(self.UAV.G * self.UAV.G.H) * self.power_factor
        # 5 reset reflecting coefficient
        """self.RIS = RIS(\
        coordinate=self.data_manager.read_init_location('RIS', 0), \
        coor_sys_z=self.data_manager.read_init_location('RIS_norm_vec', 0), \
        ant_num=16)"""
        self.RIS.Phi = np.mat(np.diag(np.ones(self.RIS.ant_num, dtype=complex)), dtype=complex)
        # 6 reset time
        self.render_obj.t_index = 0
        # 7 reset CSI
        self.H_UR.update_CSI()  # (在 CSI 更新后，重新计算整个系统中各个信道的容量，确保系统使用最新的信道条件来进行后续的优化或调度。)
        for h in self.h_U_k + self.h_U_p + self.h_R_k + self.h_R_p:
            h.update_CSI()
        # 8 reset capcaity
        self.update_channel_capacity()

    def get_communication_energy(self, delta_time=0.1 / 1000, P_max=23, P_noise=1e-3):
        total_comm_energy = 0  # 初始化总通信能耗

        # 计算用户的通信能耗
        for user in self.user_list:
            h_U_k = self.h_U_k[user.index].channel_matrix
            h_R_k = self.h_R_k[user.index].channel_matrix

            # 确保 h_U_k 和 h_R_k 是一维数组
            h_U_k = np.asarray(h_U_k).ravel()  # 展平为一维数组
            h_R_k = np.asarray(h_R_k).ravel()  # 展平为一维数组

            # 计算信道增益
            channel_gain = np.abs(h_U_k) ** 2 + np.abs(h_R_k) ** 2

            # 计算用户的SNR (假设噪声功率 P_noise 已知)
            SNR_user = channel_gain / P_noise

            # 根据信噪比调整传输功率（假设传输功率是与 SNR 成正比的）
            P_transmit_user = P_max / (1 + 1 / SNR_user)  # 简单的调整方法

            # 计算用户的通信能耗
            comm_energy_user = P_transmit_user * delta_time
            total_comm_energy += comm_energy_user

        # 计算窃听者的通信能耗
        for attacker in self.attacker_list:
            h_U_p = self.h_U_p[attacker.index].channel_matrix
            h_R_p = self.h_R_p[attacker.index].channel_matrix

            # 确保 h_U_p 和 h_R_p 是一维数组
            h_U_p = np.asarray(h_U_p).ravel()  # 展平为一维数组
            h_R_p = np.asarray(h_R_p).ravel()  # 展平为一维数组

            # 计算信道增益
            channel_gain_attacker = np.abs(h_U_p) ** 2 + np.abs(h_R_p) ** 2

            # 计算攻击者的SNR
            SNR_attacker = channel_gain_attacker / P_noise

            # 根据信噪比调整攻击者的传输功率
            P_transmit_attacker = P_max / (1 + 1 / SNR_attacker)  # 这里的调整可以根据需要修改

            # 计算攻击者的通信能耗
            comm_energy_attacker = P_transmit_attacker * delta_time
            total_comm_energy += comm_energy_attacker

        return total_comm_energy

    def step(self, action_0=0, action_1=0, G=0, Phi=0, set_pos_x=0,
             set_pos_y=0):  # (该方法是通信系统仿真中的一步操作，它的主要任务是更新系统的状态、信道、波束成形矩阵和反射系数，并根据当前的行动计算奖励值。)
        """
        test step only move UAV and update channel
        """
        # 0 update render

        global v_t
        self.render_obj.t_index += 1  # 渲染对象的时间索引
        # 1 update entities

        if self.if_move_users:  # (指定位置和速度更新用户)
            self.user_list[0].update_coordinate(0.2, -1 / 2 * math.pi)
            self.user_list[1].update_coordinate(0.2, -1 / 2 * math.pi)

        if self.if_movements:
            move_x = action_0 * self.UAV.max_movement_per_time_slot
            move_y = action_1 * self.UAV.max_movement_per_time_slot

            ######################################################
            # new for energy
            v_t = (move_x ** 2 + move_y ** 2) ** 0.5
            # self.data_manager.store_data([v_t],'velocity')
            ######################################################

            if self.reverse_x_y[0]:
                move_x = -move_x

            if self.reverse_x_y[1]:
                move_y = -move_y

            self.UAV.coordinate[0] += move_x
            self.UAV.coordinate[1] += move_y
            self.data_manager.store_data([move_x, move_y], 'UAV_movement')
        else:
            set_pos_x = map_to(set_pos_x, (-1, 1), self.border[0])
            set_pos_y = map_to(set_pos_y, (-1, 1), self.border[1])
            self.UAV.coordinate[0] = set_pos_x
            self.UAV.coordinate[1] = set_pos_y

        # 2 update channel CSI

        for h in self.h_U_k + self.h_U_p + self.h_R_k + self.h_R_p:
            h.update_CSI()
        # !!! test to make direct link zero
        if self.if_dir_link == 0:
            for h in self.h_U_k + self.h_U_p:
                h.channel_matrix = np.mat(np.zeros(shape=np.shape(h.channel_matrix)), dtype=complex)
        if self.if_with_RIS == False:
            self.H_UR.channel_matrix = np.mat(np.zeros((self.RIS.ant_num, self.UAV.ant_num)), dtype=complex)
        else:
            self.H_UR.update_CSI()
        # 3 update beamforming matrix & reflecting phase shift
        """
        self.UAV.G = G
        self.RIS.Phi = Phi
        """
        self.UAV.G = convert_list_to_complex_matrix(G, (self.UAV.ant_num, self.user_num)) * math.pow(self.power_factor,
                                                                                                     0.5)

        # fix beamforming matrix
        # self.UAV.G = np.mat(np.ones((self.UAV.ant_num, self.user_num), dtype=complex), dtype=complex) * math.pow(self.power_factor, 0.5)
        if self.if_with_RIS:
            self.RIS.Phi = convert_list_to_complex_diag(Phi, self.RIS.ant_num)
        # 4 update channel capacity in every user and attacker
        self.update_channel_capacity()
        # 5 store current system state to .mat
        self.store_current_system_sate()
        # 6 get new state
        new_state = self.observe()
        # 7 get reward
        reward = self.reward()

        # 7.1 reward with energy efficiency
        ######################################################
        if self.reward_design == 'see':
            # 获取通信总能量消耗
            total_comm_energy = self.get_communication_energy()

            # 获取原始能量消耗，并调整能量计算公式，考虑通信能量
            energy_raw = get_energy_consumption(v_t)

            # 将通信总能量消耗考虑进最小值和最大值的计算
            ENERGY_MIN = get_energy_consumption(0.25) + total_comm_energy  # 可以根据通信能耗调整最小值
            ENERGY_MAX = get_energy_consumption(0) + total_comm_energy  # 可以根据通信能耗调整最大值

            # 计算能量效率
            energy = energy_raw - ENERGY_MIN
            energy /= (ENERGY_MAX - ENERGY_MIN)

            # 确保 reward 是标量或数组都能处理
            reward = np.asarray(reward)  # 转换为 numpy 数组，以便进行逐元素处理

            # 计算能量惩罚
            energy_penalty = np.sum(
                -1 * 0.1 * np.abs(reward) * energy)  # 对 energy_penalty 进行求和  # -1 * 0.1 * reward * energy

            if reward > 0:
                reward += energy_penalty
        ######################################################

        # 8 calculate if UAV is cross the bourder     根据 UAV 是否越过边界来判断是否结束。
        reward = math.tanh(reward)  # new for energy (ori not commented)
        done = False
        x, y = self.UAV.coordinate[0:2]
        if x < self.border[0][0] or x > self.border[0][1]:
            done = True
            reward = -10
        if y < self.border[1][0] or y > self.border[1][1]:
            done = True
            reward = -10
        self.data_manager.store_data([reward], 'reward')
        return new_state, reward, done, []

    def reward(self):
        """
        used in function step to get the reward of current step
        """
        reward = 0
        reward_ = 0
        P = np.trace(self.UAV.G * self.UAV.G.H)
        if abs(P) > abs(self.UAV.G_Pmax):
            reward = abs(self.UAV.G_Pmax) - abs(P)
            reward /= self.power_factor
        else:
            for user in self.user_list:
                r = user.capacity - max(self.eavesdrop_capacity_array[:, user.index])
                if r < user.QoS_constrain:
                    reward_ += r - user.QoS_constrain
                else:
                    reward += r / (self.user_num * 2)
            if reward_ < 0:
                reward = reward_ * self.user_num * 10

        return reward
    #
    # def observe(self):  # 获取当前状态
    #     """
    #     used in function main to get current state
    #     the state is a list with
    #     """
    #     # users' and attackers' comprehensive channel （信道信息的实部虚部）
    #     comprehensive_channel_elements_list = []
    #     for entity in self.user_list + self.attacker_list:
    #         tmp_list = list(np.array(np.reshape(entity.comprehensive_channel, (1, -1)))[0])
    #         comprehensive_channel_elements_list += list(np.real(tmp_list)) + list(np.imag(tmp_list))
    #     UAV_position_list = []
    #     if self.if_UAV_pos_state:
    #         UAV_position_list = list(self.UAV.coordinate)
    #
    #     return comprehensive_channel_elements_list + UAV_position_list

    def observe(self, if_local=False):
        # 原始状态由：通道信息 + UAV位置构成
        comprehensive_channel_elements_list = []
        for entity in self.user_list + self.attacker_list:
            tmp_list = list(np.array(np.reshape(entity.comprehensive_channel, (1, -1)))[0])
            comprehensive_channel_elements_list += list(np.real(tmp_list)) + list(np.imag(tmp_list))

        UAV_position_list = list(self.UAV.coordinate) if self.if_UAV_pos_state else []

        full_state = comprehensive_channel_elements_list + UAV_position_list

        # ⚠️ 新增：屏蔽通道信息，保留UAV位置（局部观测实验）
        if self.if_local_obs:
            # 通道信息长度
            ch_len = len(comprehensive_channel_elements_list)
            local_state = [0.0] * ch_len + UAV_position_list
            return local_state

        return full_state

    def store_current_system_sate(self):  # 存储当前系统状态
        """
        function used in step() to store system state
        """
        # 1 store beamforming matrix
        row_data = list(np.array(np.reshape(self.UAV.G, (1, -1)))[0, :])
        self.data_manager.store_data(row_data, 'beamforming_matrix')
        # 2 store reflecting coefficient matrix
        row_data = list(np.array(np.reshape(diag(self.RIS.Phi), (1, -1)))[0, :])
        self.data_manager.store_data(row_data, 'reflecting_coefficient')
        # 3 store UAV state
        row_data = list(self.UAV.coordinate)
        self.data_manager.store_data(row_data, 'UAV_state')
        # 4 store user_capicity
        row_data = [user.secure_capacity for user in self.user_list] \
                   + [user.capacity for user in self.user_list]
        # 5 store G_power
        row_data = [np.trace(self.UAV.G * self.UAV.G.H), self.UAV.G_Pmax]
        self.data_manager.store_data(row_data, 'G_power')  # （波束形成矩阵的功率）
        row_data = []
        for user in self.user_list:
            row_data.append(user.capacity)
        self.data_manager.store_data(row_data, 'user_capacity')

        row_data = []
        for attacker in self.attacker_list:
            row_data.append(attacker.capacity)
        self.data_manager.store_data(row_data, 'attaker_capacity')

        row_data = []
        for user in self.user_list:
            row_data.append(user.secure_capacity)
        self.data_manager.store_data(row_data, 'secure_capacity')

    def update_channel_capacity(self):
        """
        function used in step to calculate user and attackers' capacity
        """
        # 1 calculate eavesdrop rate
        for attacker in self.attacker_list:
            attacker.capacity = self.calculate_capacity_array_of_attacker_p(attacker.index)
            self.eavesdrop_capacity_array[attacker.index, :] = attacker.capacity
            # remmeber to update comprehensive_channel
            attacker.comprehensive_channel = self.calculate_comprehensive_channel_of_attacker_p(attacker.index)
        # 2 calculate unsecure rate
        for user in self.user_list:
            user.capacity = self.calculate_capacity_of_user_k(user.index)
            # 3 calculate secure rate
            user.secure_capacity = self.calculate_secure_capacity_of_user_k(user.index)
            # remmeber to update comprehensive_channel
            user.comprehensive_channel = self.calculate_comprehensive_channel_of_user_k(user.index)

    def calculate_comprehensive_channel_of_attacker_p(self, p):
        """
        used in update_channel_capacity to calculate the comprehensive_channel of attacker p
        """
        h_U_p = self.h_U_p[p].channel_matrix
        h_R_p = self.h_R_p[p].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_p).H * self.H_UR.channel_matrix
        return h_U_p.H + Psi.H * H_c

    def calculate_comprehensive_channel_of_user_k(self, k):
        """
        used in update_channel_capacity to calculate the comprehensive_channel of user k
        """
        h_U_k = self.h_U_k[k].channel_matrix
        h_R_k = self.h_R_k[k].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_k).H * self.H_UR.channel_matrix
        return h_U_k.H + Psi.H * H_c

    def calculate_capacity_of_user_k(self, k):
        """
        function used in update_channel_capacity to calculate one user
        """
        noise_power = self.user_list[k].noise_power
        h_U_k = self.h_U_k[k].channel_matrix
        h_R_k = self.h_R_k[k].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_k).H * self.H_UR.channel_matrix
        G_k = self.UAV.G[:, k]
        G_k_ = 0
        if len(self.user_list) == 1:
            G_k_ = np.mat(np.zeros((self.UAV.ant_num, 1), dtype=complex), dtype=complex)
        else:
            G_k_1 = self.UAV.G[:, 0:k]
            G_k_2 = self.UAV.G[:, k + 1:]
            G_k_ = np.hstack((G_k_1, G_k_2))
        alpha_k = math.pow(abs((h_U_k.H + Psi.H * H_c) * G_k), 2)
        beta_k = math.pow(np.linalg.norm((h_U_k.H + Psi.H * H_c) * G_k_), 2) + dB_to_normal(noise_power) * 1e-3
        return math.log10(1 + abs(alpha_k / beta_k))

    def calculate_capacity_array_of_attacker_p(self, p):
        """
        function used in update_channel_capacity to calculate one attacker capacities to K users
        output is a K length np.array ,shape: (K,)
        """
        K = len(self.user_list)
        noise_power = self.attacker_list[p].noise_power
        h_U_p = self.h_U_p[p].channel_matrix
        h_R_p = self.h_R_p[p].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_p).H * self.H_UR.channel_matrix
        if K == 1:
            G_k = self.UAV.G
            G_k_ = np.mat(np.zeros((self.UAV.ant_num, 1), dtype=complex), dtype=complex)
            alpha_p = math.pow(abs((h_U_p.H + Psi.H * H_c) * G_k), 2)
            beta_p = math.pow(np.linalg.norm((h_U_p.H + Psi.H * H_c) * G_k_), 2) + dB_to_normal(noise_power) * 1e-3
            return np.array([math.log10(1 + abs(alpha_p / beta_p))])
        else:
            result = np.zeros(K)
            for k in range(K):
                G_k = G_k = self.UAV.G[:, k]
                G_k_1 = self.UAV.G[:, 0:k]
                G_k_2 = self.UAV.G[:, k + 1:]
                G_k_ = np.hstack((G_k_1, G_k_2))
                alpha_p = math.pow(abs((h_U_p.H + Psi.H * H_c) * G_k), 2)
                beta_p = math.pow(np.linalg.norm((h_U_p.H + Psi.H * H_c) * G_k_), 2) + dB_to_normal(noise_power) * 1e-3
                result[k] = math.log10(1 + abs(alpha_p / beta_p))
            return result

    def calculate_secure_capacity_of_user_k(self, k=2):
        """
        function used in update_channel_capacity to calculate the secure rate of user k
        """
        user = self.user_list[k]
        R_k_unsecure = user.capacity
        R_k_maxeavesdrop = max(self.eavesdrop_capacity_array[:, k])
        secrecy_rate = max(0, R_k_unsecure - R_k_maxeavesdrop)
        return secrecy_rate

    def get_system_action_dim(self):  # 确定系统动作维度
        """
        function used in main function to get the dimention of actions
        """
        result = 0
        # 0 UAV movement
        result += 2
        # 1 RIS reflecting elements
        if self.if_with_RIS:
            result += self.RIS.ant_num
        else:
            result += 0
        # 2 beamforming matrix dimention
        result += 2 * self.UAV.ant_num * self.user_num
        return result

    def get_system_state_dim(self):
        """
        function used in main function to get the dimention of states
        """
        result = 0
        # users' and attackers' comprehensive channel
        result += 2 * (self.user_num + self.attacker_num) * self.UAV.ant_num
        # UAV position
        if self.if_UAV_pos_state:
            result += 3
        return result
