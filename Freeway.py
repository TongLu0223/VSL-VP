#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：FREEWAY
@File    ：Freeway.py
@Author  ：Lu
@Date    ：2023/11/2 10:25 
'''
import queue
import random
from collections import deque
import traci
import tensorflow as tf
import tensorlayer as tl
import sys
sys.path.append('H:\Lu_programme\programme')
import numpy as np
import plexe
from plexe import Plexe
from plexe import ACC, CACC, FAKED_CACC
from datetime import  datetime
from ccparams import *
from utils import *

class Freeway:
    def __init__(self):
        "Init Environment"

        # Network
        self.Q_network = self.get_model(input_dim = 27, output_dim = 4)
        self.Q_network.train()
        self.target_Q_network = self.get_model(input_dim = 27, output_dim = 4)
        self.target_Q_network.eval()

        # Train
        self.episode = 0
        self.episode_max = 1
        self.epsilon = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.05

        self.memory = deque(maxlen = 200000)
        self.batch = 256
        self.gamma = 0.95
        self.learn_rate = 1e-3
        self.opt = tf.optimizers.Adam(self.learn_rate) # 优化器

        # Simulation
        self.simulation_step = 30000  # 10 -> 1s
        self.step = 0
        self.rend = True

        self.leader_vids = []
        self.collision_vids = []
        self.start_vids = []
        self.windows = queue.Queue(maxsize=6)
        self.window_cav_vids = []
        self.window_done_vids = []

        self.plexe = None
        self.join_state = dict()
        self.change_lane_state = dict()
        self.topology = dict()

        self.first_action = True
        self.collision = False
        self.done = False

        # Road parameter
        self.lane_ids = ['E0', 'E1', 'E2']
        self.vsl_lane = ['E1']
        self.start_platoon_lane = 'E2'

        self.platoon_lane = -1
        self.accident_lane = -1
        self.accident_pos = -1
        self.R = 10000  # CAV 通信距离

        # Output data
        self.total_reward = 0
        self.total_loss = []

        self.file_path = 'output/' + datetime.now().strftime('%y_%m_%d')

    def reinit(self):
        "每回合重置参数"
        # Simulation
        self.step = 0

        self.leader_vids = []
        self.collision_vids = []
        self.start_vids = []
        self.windows.queue.clear()
        self.window_cav_vids = []

        self.join_state = dict()
        self.change_lane_state = dict()
        self.topology = dict()

        self.first_action = True
        self.collision = False
        self.done = False

        # Road parameter
        self.platoon_lane = -1
        self.accident_lane = -1
        self.accident_pos = -1

        # Output data
        self.total_reward = 0
        self.total_loss = []

    def update_epsilon(self):
        "更新epsilon, 除非epsilon已经足够小"
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_model(self, input_dim, output_dim):
        "获取模型"
        # DQN
        self.input = tl.layers.Input(shape=(None, input_dim))
        self.h1 = tl.layers.Dense(64, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(self.input)
        self.h2 = tl.layers.Dense(32, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(self.h1)
        self.output = tl.layers.Dense(output_dim, act=None, W_init=tf.initializers.GlorotUniform())(self.h2)

        return tl.models.Model(inputs=self.input, outputs=self.output)

    def get_platoon_intensity(self):
        "获取每条车道上的车队强度"
        # 获取每个车道上的 id_list = [0, 1, 0]; 0 for CAV, 1 for HDV, 从左到右
        platoon_intensity = [0, 0, 0]
        lane_ids = self.lane_ids
        for i in range(3):
            id_list = []
            vehicle_id = []
            for x in range(3):
                vehicle_id.append(traci.lane.getLastStepVehicleIDs(lane_ids[x] + '_' + str(i)))
            vehicle_id = [item for sublist in vehicle_id for item in sublist]
            # print(vehicle_id)
            total = 0  # 总共CAV的数量
            for j in vehicle_id:
                if j[2] == '0':
                    id_list.append(1)
                    total += 1
                elif j[2] == '1':
                    id_list.append(0)
            # print(id_list) # 从左到右, 1 for CAV, 0 for HV
            length = len(id_list)
            count = 0  # 单个CAV的数量
            for j in range(length):
                if id_list[j] == 1 \
                        and (j == 0 or id_list[j - 1] == 0) \
                        and (j == length - 1 or id_list[j + 1] == 0):
                    count += 1
            # print('---',count, total)
            if total == 0:
                platoon_intensity[i] = 0.0
            else:
                platoon_intensity[i] = (total - count) / total
        print("车队强度：", platoon_intensity)
        return platoon_intensity

    def get_max_vid_value(self):
        """
        获取当前存在车辆的最大id, 为了之后创建车辆
        """
        current_vids = traci.vehicle.getIDList()

        max_cav_value = -1
        max_hdv_value = -1
        max_cav_id = None
        max_hdv_id = None

        for vid in current_vids:
            if vid.startswith('v.0.'):
                cav_value = int(vid.split('.')[-1])
                if cav_value > max_cav_value:
                    max_cav_value = cav_value
                    max_cav_id = vid
            elif vid.startswith('v.1.'):
                hdv_value = int(vid.split('.')[-1])
                if hdv_value > max_hdv_value:
                    max_hdv_value = hdv_value
                    max_hdv_id = vid

        return max_cav_value, max_hdv_value

    def get_random_matrix(self, rows, cols):
        '''
        生成随机的CAV类型
        大小为 rows * cols
        要求每一行中必须都有 0->CAV
        且CAV总数要占百分之60
        '''
        matrix = []

        for i in range(rows):
            row = [1] * cols  # 先将整行初始化为1（HDV）
            random_index = random.randint(0, cols - 1)
            row[random_index] = 0  # 随机选择一个位置设置为0（CAV）
            matrix.append(row)

        # 随机选择一些位置设置为0（CAV），以满足总数的60%要求
        num_zeros = int(rows * cols * 0.6) - rows
        zero_positions = [(i, j) for i in range(rows) for j in range(cols) if matrix[i][j] == 1]
        random.shuffle(zero_positions)

        prev_zero_row = -1
        prev_zero_col = -1

        for i in range(min(num_zeros, len(zero_positions))):
            row, col = zero_positions[i]

            # 检查前一个位置是否已经是0，如果是的话，选择下一个位置
            if row == prev_zero_row and col == prev_zero_col:
                i += 1
                row, col = zero_positions[i]

            matrix[row][col] = 0
            prev_zero_row = row
            prev_zero_col = col

        return matrix

    def get_random_pos(self, v_num=30, distance=30):
        "产生随机的车辆位置, v_num = 3 * n"
        result = []
        a = random.randint(3, int(distance/2))
        b = random.randint(3, int(distance/2))
        c = random.randint(3, int(distance/2))
        for i in range(int(v_num / 3)):
            row = [a, b, c]
            result.append(row)
            a = a + random.randint(distance, distance + 10)
            b = b + random.randint(distance, distance + 10)
            c = c + random.randint(distance, distance + 10)

        # print(result)
        return result

    def get_random_vtype(self, v_num=30, p=0.6):
        "随机生成车辆类型, v_num = 3 * n, p = [0.2, 0.4, 0.6, 0.8, 1.0]"
        random_vtype = []
        cav_num = int(v_num * p)
        hav_num = v_num - cav_num

        # 初始化列表
        result = [0] * cav_num + [1] * hav_num
        random.shuffle(result)

        # 检查是否存在连续三个1，若有则重新生成
        while '111' in ''.join(map(str, result)):
            random.shuffle(result)

        while '000000' in ''.join(map(str, result)):
            random.shuffle(result)

        for i in range(int(v_num / 3)):
            row = result[i * 3: (i + 1) * 3]
            random_vtype.append(row)

        return random_vtype

    def get_vid_posX(self, vid):
        "获取每个车辆的x位置, pos[0] -> lane_position + 之前的 lane length"
        pos = traci.vehicle.getPosition(vid)
        return pos[0]

    def get_total_vids(self):
        "获取仿真中所有车辆的ID, 按位置进行排序"

        vids = traci.vehicle.getIDList()
        vids = [item for item in vids if not item.startswith('v.2.')]
        vids = sorted(vids, key=self.get_vid_posX)
        self.total_vids = vids

        return vids

    def get_not_window_vids(self):
        "获取未参与分组的车辆ID, platoon_area BRS-2"

        vids = self.get_total_vids()
        platoon_area_vids = []
        for vid in vids:
            pos = traci.vehicle.getPosition(vid)[0]
            if pos > 1200 and pos < self.accident_pos and vid not in self.window_done_vids:
                platoon_area_vids.append(vid)

        # platoon_area_vids.reverse()
        return platoon_area_vids

    def get_lane_vids(self, lane_id):
        "获取该路段上车辆ID, 按位置进行排序"
        vids = []
        for lane_index in range(3):
            id = lane_id + '_' + str(lane_index)
            vids.append(traci.lane.getLastStepVehicleIDs(id))
        vids = [item for sublist in vids for item in sublist]
        # print(vids)
        vids = sorted(vids, key=self.get_vid_posX)
        # print(vids)
        vids = [x for x in vids if x[2] != '2']
        return vids

    def get_state(self, vids, leader_id):
        "根据vids获取车辆的状态, 状态为窗口内所有车辆的id, type, pos, lane, speed + 瓶颈lane, pos"
        state = [[int(leader_id[4]),
                  int(0),
                  round(traci.vehicle.getPosition(leader_id)[0], 2),
                  traci.vehicle.getLaneIndex(leader_id),
                  round(traci.vehicle.getSpeed(leader_id), 2)]]
        for id in vids:
            type = int(id[2])
            pos = round(traci.vehicle.getPosition(id)[0], 2)
            lane_index = traci.vehicle.getLaneIndex(id)
            speed = round(traci.vehicle.getSpeed(id), 2)
            s = [int(id[4]), type, pos, lane_index, speed]
            state.append(s)

        state.append([self.accident_lane, self.accident_pos])

        flattened_state = [item for sublist in state for item in sublist]

        return flattened_state

    def get_action(self, state, agent_ids):
        '''
        根据状态, 获取动作
        先根据状态判断输出动作的维度, 再调用模型
        '''
        action_size = len(agent_ids)
        print('action_size:', action_size)
        a = 0
        if np.random.rand() >= self.epsilon:
            print('---非随机动作')
            if action_size == 2:
                q = self.Q_network(np.array(state, dtype='float32').reshape([-1, 27]))
                a = np.argmax(q)
            # elif action_size == 3:
            #     q = ...
            #     a = np.argmax(q)
            # elif action_size == 4:
            #     q = ...
            #     a = np.argmax(q)
            # elif action_size == 1:
            #     q = ...
            #     a = np.argmax(q)
        else:
            a = np.random.randint(0, 2**action_size - 1)

        # print('a:', a)
        action = self.decimal_to_binary_list(a, num_bits = action_size)

        return action

    def get_avg_speed(self, vids):
        '''
        获取所有CAV的平均速度
        '''
        total_speed = 0
        for id in vids:
            total_speed += round(traci.vehicle.getSpeed(id), 2)
        avg_speed = total_speed / len(vids)
        return round(avg_speed, 2)

    def get_last_vid(self, vids):
        """
        获取最靠后vid的id以及位置
        """
        last_pos = None
        last_vid = None
        for id in vids:
            pos = traci.vehicle.getPosition(id)[0]
            if last_pos is None or pos < last_pos:
                last_vid = id
                last_pos = pos
        return last_vid, last_pos

    def get_reward(self, cav_vids):
        '''
        获取奖励值
        1) CAV的平均速度 speed
        2) 是否发生了碰撞
        3) 是否通过了瓶颈
        4) 是否完成了合并
        '''

        # r1 CAV average speed
        if len(cav_vids) == 0:
            r1 = 0
        else:
            r1 = self.get_avg_speed(cav_vids)

        # r2 whether 发生 collision
        if self.collision:
            r2 = COLLISION_REWARD
        else:
            r2 = 0

        # r3 whether pass the bottleneck area
        last_vid, last_pos = self.get_last_vid(cav_vids)
        if last_pos > self.accident_pos:
            print('The last CAV have passed the bottleneck area')
            r3 = 0
            pass_done = True
        else:
            r3 = TIME_REWARD
            pass_done = False

        # r4 is the ratio of CAVs perform platooning
        total_count = len(cav_vids)
        index_count = 0
        for cav in cav_vids:
            if cav in self.join_state:
                index_count += 1

        if total_count == 0:
            r4 = 0
        else:
            r4 = index_count / total_count

        if pass_done:
            r4 = 0
        # print('r4:', r4)

        if r4 == 1.0:
            r4 = 2

        reward = r1 + r2 + r4 * 40
        return reward

    def get_reward_shaping(self, cav_vids, leader):
        '''
        获取标准化后的奖励
        '''
        # r0 -> the speed of CAVs passing bottleneck area
        cavs = cav_vids[:]
        cavs.append(leader)
        print('cav_vids: ', cavs)
        if self.done == True:
            avg_speed = self.get_avg_speed(cavs)
            r0 = self.normalize_speed(avg_speed)
            print('r0 = ', r0)
        else:
            r0 = 0

        # r1 -> safety
        if self.collision:
            r1 = COLLISION_REWARD
        else:
            r1 = 0
        # r2 -> platooning density
        total_count = len(cav_vids)
        index_count = 0
        for cav in cav_vids:
            if cav in self.join_state:
                index_count += 1

        if total_count == 0:
            r2 = 0
        else:
            r2 = index_count / total_count

        r = r0 + r1 + r2 * 0.1
        return r

    def get_front_lane_vid(self, vid):
        '''
        变道完成后, 获取同车道前车 cav vid
        '''
        lane_id = traci.vehicle.getLaneID(vid)
        # print('lane_id:', lane_id)
        # lane_id = 'E2_' + str(self.platoon_lane)

        vids = list(traci.lane.getLastStepVehicleIDs(lane_id))
        # 加上类型
        vids = [item for item in vids if item.startswith('v.0.')]
        index = vids.index(vid)

        if index+1 == len(vids):
            return vids[index]

        return vids[index + 1]

    def get_front_leader_vid(self, vid):
        """
        变道完成后, 获取前方最近的leader vid
        """
        lane_id = traci.vehicle.getLaneID(vid)


        vids = list(traci.lane.getLastStepVehicleIDs(lane_id))
        # 加上类型
        vids = [item for item in vids if item.startswith('v.0.')]
        index = vids.index(vid)

        for i in range(index, len(vids)):
            if vids[i] in self.leader_vids:
                return vids[i]

    def get_front_vid(self, vid):
        "获取该车辆前方最近的vid (可以不同车道)"
        vids = self.get_total_vids()
        cav_vids = [item for item in vids if item.startswith('v.0.')]

        index = cav_vids.index(vid)
        return cav_vids[index + 1]

    def get_window_cav_vids(self, vids):
        '''
        从window里面获取所要控制的agent的id
        还要去除窗口内所有的leader
        '''
        cav_vids = [item for item in vids if item.startswith('v.0.')]

        intersection = set(cav_vids) & set(self.leader_vids)

        cav_vids = [x for x in cav_vids if x not in intersection]

        # cav_vids.reverse()
        # if cav_vids not in self.window_cav_vids:
        #     self.window_cav_vids.append(cav_vids)
        return cav_vids

    def select_platoon_lane(self):
        "选择专用车道"
        '''
        算法流程: 获取每个车道上的车辆数目以及HV的数目, 求出CAV所占该车道总车辆数的比例,
        选择比例最小的车道, 若比例相同, 优先最右侧车道
        其他思路: 可以设置为某一个范围内的车辆数, 而不是整个车道上的车辆数目
        '''
        # 选取车排车道
        # 获取每个车道的车辆数
        if self.platoon_lane >= 0:
            return
        lane_ids = self.lane_ids
        vehicleCount = [-1, -1, -1]
        vehicleCount_cav = [0, 0, 0]
        P_cav = [0, 0, 0]
        for i in range(3):
            # 获取每个车道上车辆数目
            # vehicleCount[i] = traci.lane.getLastStepVehicleNumber(lane_id[i])
            # 获取每个车道上HV车辆的数目
            vehicle_id = []
            for x in range(3):
                vehicle_id.append(traci.lane.getLastStepVehicleIDs(lane_ids[x] + '_' + str(i)))
            vehicle_id = [item for sublist in vehicle_id for item in sublist]
            # print(vehicle_id)
            count = 0
            total = 0
            for id in vehicle_id:
                if id[2] == '0':
                    count += 1
                    total += 1
                elif id[2] == '1':
                    total += 1
            vehicleCount[i] = total
            vehicleCount_cav[i] = count
            # 计算每个车道中HV所占的比例
            if i == int(self.accident_lane):
                P_cav[i] = -1000.0
            elif vehicleCount[i] == 0:
                P_cav[i] = 1000.0
            else:
                P_cav[i] = vehicleCount_cav[i] / vehicleCount[i]

        # print(vehicleCount)
        # print(vehicleCount_cav)
        print("CAV市场渗透率:", P_cav)
        platoon_intensity = self.get_platoon_intensity()
        Max_P = max(P_cav)
        # print(Max_P)
        # 找到数组中最大值的索引（全部）
        max_indices = [i for i, x in enumerate(P_cav) if x == Max_P]
        platoon_lane = -1
        if len(max_indices) == 1:
            platoon_lane = np.argmax(P_cav)
        elif len(max_indices) == 2:
            if platoon_intensity[max_indices[0]] >= platoon_intensity[max_indices[1]]:
                platoon_lane = max_indices[0]
            else:
                platoon_lane = max_indices[1]
        # # 设置好车排车道

        platoon_lane_id = self.start_platoon_lane + '_' + str(platoon_lane)
        traci.lane.setDisallowed(laneID=platoon_lane_id, disallowedClasses=['bus'])

        # platoon_lane = 2
        print('车排车道已选好, ID为: ', end="")
        print(platoon_lane)
        self.platoon_lane = platoon_lane
        # 其他思路: 获取一定范围内的车道数

    def select_leader(self, vids):
        """
        从vids中获取leader, 选择窗口内位置最靠前的CAV
        """
        leader = 'v.0.0'

        # 先判断该window里面是否有leader, 有就不选
        set_vids = set(vids)
        set_leader_vids = set(self.leader_vids)
        if set_vids & set_leader_vids:
            print('该分组已经选择过leader了, leader为:', set_vids & set_leader_vids)
            return list(set_vids & set_leader_vids)[-1]

        # 选择位置最靠前的CAV
        for id in vids:
            if id[2] == '0' and traci.vehicle.getLaneIndex(id) == self.platoon_lane:
                # 确保 id 没有执行过加入车队的命令
                if not self.is_vid_in_join_state(id):
                    leader = id
                    break
            elif id[2] == '0':
                # 确保 id 没有执行过加入车队的命令
                if not self.is_vid_in_join_state(id):
                    leader = id
                    break
        print('leader:', leader)

        if leader == 'v.0.0':
            print('该窗口内没有leader')
            return leader

        # leader 设置
        traci.vehicle.setColor(typeID=leader, color=((160, 32, 240)))

        self.plexe.set_path_cacc_parameters(vid=leader, distance=5, xi=2, omega_n=1, c1=0.5)
        # traci.vehicle.setLaneChangeMode(leader, 256)
        # traci.vehicle.changeLane(vehID=leader, laneIndex=self.platoon_lane, duration=0)
        self.plexe.set_fixed_lane(vid=leader, lane=self.platoon_lane, safe=True)

        traci.vehicle.setMaxSpeed(leader, MAX_SPEED)
        self.plexe.set_cc_desired_speed(leader, MAX_SPEED)
        self.plexe.set_acc_headway_time(leader, 1.5)

        self.plexe.use_controller_acceleration(leader, False)
        self.plexe.set_active_controller(vid=leader, controller=ACC)

        self.leader_vids.append(leader)

        return leader

    def clear_platoon_lane(self):
        "清理车队的专用车道, 尽量不让HDV在此车道上行驶"
        lane_ids = ['E3', 'E2']
        # lane_ids = ['E2']
        for id in lane_ids:
            lane_id = id + '_' + str(self.platoon_lane)
            platoon_lane_vid = traci.lane.getLastStepVehicleIDs(laneID=lane_id)
            for vid in platoon_lane_vid:
                if vid[2] == '1':
                    traci.vehicle.changeLane(vehID=vid, laneIndex=(self.platoon_lane + 1) % 2, duration=2.0)

    def create_accident(self):
        "创建瓶颈问题, 即一些发生事故的拥堵车辆"

        parking_id = 'ParkAreaA'
        accident_pos = traci.parkingarea.getStartPos(parking_id)
        lane_id = traci.parkingarea.getLaneID(parking_id)
        lane_index = lane_id[3]
        for i in range(5):
            vid = 'v.2.%d' % i
            traci.vehicle.add(vehID=vid, routeID='accident_route', departPos=str(accident_pos + (i + 1) * 10),
                              departLane=lane_index, departSpeed=str(0.00))
            traci.vehicle.setParkingAreaStop(vehID=vid, stopID=parking_id, duration=100000)
            traci.vehicle.setColor(vid,(255, 0, 0))
        self.accident_lane = lane_index
        self.accident_pos = accident_pos + 1000

    def create_vehicles(self):
        "添加车辆, 固定的, 没有随机性, 9辆CAV, 6辆HDV"

        max_cav_value, max_hdv_value = self.get_max_vid_value()
        if max_cav_value == -1:
            max_cav_value = 0
        if max_hdv_value == -1:
            max_hdv_value = 0
        # print('max_cav_value:', max_cav_value, 'max_hdv_value:', max_hdv_value)

        cav_pos = [10, 3, 30, 53, 64, 60, 88, 105, 130]
        cav_index = [1, 2, 0, 2, 0, 1, 1, 2, 0]
        hdv_pos = [4, 33, 40, 85, 100, 125]
        hdv_index = [0, 2, 1, 2, 0, 1]

        for i in range(len(cav_pos)):
            cav_vid = 'v.0.' + str(max_cav_value + i + 1)
            pos = cav_pos[i]
            index = cav_index[i]
            traci.vehicle.add(vehID=cav_vid, routeID="highway_route", typeID="vtypeauto",
                              departLane= index, departPos=pos, departSpeed=20.00)

        for j in range(len(hdv_pos)):
            hdv_vid = 'v.1.' + str(max_hdv_value + j + 1)
            pos = hdv_pos[j]
            index = hdv_index[j]
            traci.vehicle.add(vehID=hdv_vid, routeID="highway_route", typeID="passenger",
                              departLane=index, departPos=pos, departSpeed=20.00)

    def create_vehicles_random(self):
        """
        带有随机性创建车辆, 位置车道固定, 但车辆类型不确定
        num = 15, CAV占60%
        """
        max_cav_value, max_hdv_value = self.get_max_vid_value()
        if max_cav_value == -1:
            max_cav_value = 0
        if max_hdv_value == -1:
            max_hdv_value = 0
        # print(max_cav_value, max_hdv_value)
        # position = [[3, 4, 10], [30, 33, 40], [53, 60, 64], [85, 88, 100], [105, 100, 125]]
        # lane_index = [[2, 1, 0], [0, 2, 1], [2, 1, 0], [2, 1, 0], [2, 0, 1]]
        position = [[3, 4, 8], [33, 34, 40], [63, 66, 74], [95, 107, 100], [140, 130, 135]]
        lane_index = [[2, 1, 0], [0, 2, 1], [2, 1, 0], [2, 1, 0], [2, 0, 1]]
        # vtype = [[1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0]]
        # vtype = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 1]]
        vtype = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0]]
        vtype = self.get_random_matrix(5, 3)
        #
        print(vtype)
        for i in range(len(position)):
            for j in range(len(position[i])):
                pos = position[i][j]
                index = lane_index[i][j]
                type = vtype[i][j]
                if type == 0:
                    vid = 'v.0.' + str(1 + max_cav_value)
                    max_cav_value += 1
                    traci.vehicle.add(vehID=vid, routeID="highway_route", typeID="vtypeauto",
                                      departLane=index, departPos=pos, departSpeed=20.00)
                elif type == 1:
                    vid = 'v.1.' + str(1 + max_hdv_value)
                    max_hdv_value += 1
                    traci.vehicle.add(vehID=vid, routeID="highway_route", typeID="passenger",
                                      departLane=index, departPos=pos, departSpeed=20.00)

    def create_vehicles_new(self, v_num=30, CAV_penetration_rate=0.6):
        "按照不同的CAV渗透率以及间距生成车辆 [20%, 40%, 60%, 80%, 100%]"
        # 获取下一个生成车辆的ID
        max_cav_value, max_hdv_value = self.get_max_vid_value()
        if max_cav_value == -1:
            max_cav_value = 0
        if max_hdv_value == -1:
            max_hdv_value = 0

        # random
        # positions = self.get_random_pos(v_num, distance = 30)
        # vtypes = self.get_random_vtype(v_num, p = CAV_penetration_rate)

        # define
        positions = [[4, 3, 6], [38, 43, 46], [74, 80, 81], [113, 115, 120], [145, 145, 156], [176, 176, 194], [206, 213, 232], [237, 250, 267], [272, 289, 297], [303, 324, 331]]
        vtypes = [[0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 1], [0, 1, 0]]

        for i in range(len(positions)):
            pos = positions[i]
            vtype = vtypes[i]
            for j in range(len(pos)):
                if vtype[j] == 0:
                    vid = 'v.0.' + str(1 + max_cav_value)
                    max_cav_value += 1
                    traci.vehicle.add(vehID=vid, routeID="highway_route", typeID="vtypeauto",
                                      departLane=j, departPos=pos[j], departSpeed=20.00)
                elif vtype[j] == 1:
                    vid = 'v.1.' + str(1 + max_hdv_value)
                    max_hdv_value += 1
                    traci.vehicle.add(vehID=vid, routeID="highway_route", typeID="passenger",
                                      departLane=j, departPos=pos[j], departSpeed=20.00)


        ''' new version'''
        # positions = [[5, 17, 30, 40, 53], [70, 82, 95, 110, 115], [134, 142, 147, 170, 175]]
        # vtypes = [[1, 0, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 0, 0]]
        # lanes = [[1, 2, 0, 1, 2], [1, 2, 1, 0, 2], [0, 2, 1, 0, 2]]

        # positions = [[5, 17, 30, 40, 53]]
        # vtypes = [[1, 0, 0, 1, 0]]
        # lanes = [[1, 2, 0, 1, 2]]
        #
        # print("pos:", positions)
        # print("vtype:", vtypes)
        #
        # for i in range(len(positions)):
        #     pos = positions[i]
        #     vtype = vtypes[i]
        #     lane = lanes[i]
        #     for j in range(len(pos)):
        #         if vtype[j] == 0:
        #             vid = 'v.0.' + str(1 + max_cav_value)
        #             max_cav_value += 1
        #             traci.vehicle.add(vehID=vid, routeID="highway_route", typeID="vtypeauto",
        #                               departLane=lane[j], departPos=pos[j], departSpeed=20.00)
        #         elif vtype[j] == 1:
        #             vid = 'v.1.' + str(1 + max_hdv_value)
        #             max_hdv_value += 1
        #             traci.vehicle.add(vehID=vid, routeID="highway_route", typeID="passenger",
        #                               departLane=lane[j], departPos=pos[j], departSpeed=20.00)
                    # traci.vehicle.setLaneChangeMode(vid, 256)

    def create_vehicles_new_v2(self, v_num=30, p=0.6):
        "按照不同的CAV渗透率生成车辆"
        # 获取下一个生成车辆
        max_cav_value, max_hdv_value = self.get_max_vid_value()
        if max_cav_value == -1:
            max_cav_value = 0
        if max_hdv_value == -1:
            max_hdv_value = 0

        positions = [[5, 17, 30, 40, 53], [70, 90, 95, 110, 115],
                     [134, 142, 147, 170, 175], [193, 207, 225, 227, 263]]
        # positions = [[5, 37, 50, 60, 73], [90, 110, 115, 130, 135],
        #              [154, 162, 167, 190, 195], [213, 227, 245, 247, 283]]

        # 1 for HDV, 0 for CAV
        # 0.0
        # vtypes = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
        #           [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]

        # 0.2
        # vtypes = [[1, 1, 0, 1, 1], [0, 1, 1, 1, 1],
        #           [1, 1, 1, 1, 0], [1, 1, 1, 0, 1]]

        # 0.4
        # vtypes = [[1, 0, 1, 1, 0], [0, 1, 1, 0, 1],
        #           [1, 0, 1, 0, 1], [0, 1, 0, 1, 1]]

        # 0.6
        # vtypes = [[1, 0, 0, 1, 0], [0, 1, 0, 1, 0],
        #           [0, 1, 1, 0, 0], [1, 0, 0, 0, 1]]

        # 0.7
        # vtypes = [[1, 0, 0, 0, 0], [0, 1, 0, 1, 0],
        #           [0, 1, 0, 0, 0], [1, 0, 0, 0, 1]]

        # 0.8
        # vtypes = [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
        #           [1, 0, 0, 0, 0], [0, 0, 0, 0, 1]]

        # 1.0
        vtypes = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

        lanes = [[1, 2, 0, 1, 2], [1, 2, 1, 0, 2],
                 [0, 2, 1, 0, 2], [1, 2, 0, 1, 1]]

        # positions = [[5, 17, 30, 40, 53]]
        # vtypes = [[1, 0, 0, 1, 0]]
        # lanes = [[1, 2, 0, 1, 2]]

        print("pos:", positions)
        print("vtype:", vtypes)

        for i in range(len(positions)):
            pos = positions[i]
            vtype = vtypes[i]
            lane = lanes[i]
            for j in range(len(pos)):
                if vtype[j] == 0:
                    vid = 'v.0.' + str(1 + max_cav_value)
                    max_cav_value += 1
                    traci.vehicle.add(vehID=vid, routeID="highway_route", typeID="vtypeauto",
                                      departLane=lane[j], departPos=pos[j], departSpeed=20.00)
                elif vtype[j] == 1:
                    vid = 'v.1.' + str(1 + max_hdv_value)
                    max_hdv_value += 1
                    traci.vehicle.add(vehID=vid, routeID="highway_route", typeID="passenger",
                                      departLane=lane[j], departPos=pos[j], departSpeed=20.00)

    def vsl(self):
        "开启VSL区域限制"
        vids = self.get_total_vids()
        for id in vids:
            lane_id = traci.vehicle.getLaneID(id)
            lane = lane_id[:2]
            if lane in self.vsl_lane:
                traci.vehicle.setMaxSpeed(id, speed = 15.00)
            else:
                if id[2] == '0':
                    if id not in self.leader_vids:
                        traci.vehicle.setMaxSpeed(id, speed = DESIRED_SPEED)
                elif id[2] == '1':
                    traci.vehicle.setMaxSpeed(id, speed = DESIRED_SPEED)

    def sliding_windows(self, N, R):
        """
        滑动窗口算法
        N -> max num of vehicles within a window
        R -> max CAV communication distance
        1) 动态窗口: 每次执行编队算法的时候都执行窗口分组
        2) 静态窗口: 分完一次组后不再更新
        """
        windows = []
        current_window = []
        vids = self.get_total_vids()
        # print('total_vids:', vids)
        # 只在瓶颈问题前划分窗口
        for id in vids[::-1]:
            pos = traci.vehicle.getPosition(id)[0]
            if pos > self.accident_pos:
                vids.remove(id)
            else:
                break

        # print('total_vids_before_window:', vids)

        # print('total_vids:', vids)
        for vid in vids[::-1]:
            vtype = vid[2]
            pos = traci.vehicle.getPosition(vid)[0]
            if not current_window:
                if vtype == '0':
                    current_window.append(vid)
                else:
                    current_window.append(vid)
            elif len(current_window) < N:
                if vtype == '0':
                    last_cav_id = '00'
                    for id in current_window:
                        if id[2] == '0':
                            last_cav_id = id
                    if last_cav_id != '00':
                        last_cav_pos = traci.vehicle.getPosition(last_cav_id)[0]
                        if abs(last_cav_pos - pos) <= R:
                            current_window.append(vid)
                        else:
                            windows.append(current_window)
                            current_window = []
                            current_window.append(vid)
                    else:
                        current_window.append(vid)
                else:
                    current_window.append(vid)

            if len(current_window) == N:
                windows.append(current_window)
                current_window = []

        if current_window:
            windows.append(current_window)

        return windows

    def sliding_windows_new(self, N, R):
        """
        滑动窗口算法
        N -> max num of vehicels within a window
        R -> max CAV communication distance
        动态窗口, 截取已经进入E2且在瓶颈区域前所有车辆划分窗口
        """
        windows = []
        current_window = []
        vids = self.get_total_vids()
        # vids = self.get_not_window_vids()

        # 从前往后访问
        for vid in vids[::-1]:
            vtype = vid[2] # 'v.0.n' -> CAV
            pos = traci.vehicle.getPosition(vid)[0]
            # if pos > self.accident_pos or pos < 1000:
            #     continue
            if pos < 1000:
                continue
            if not current_window:  # 为空
                current_window.append(vid)
            elif len(current_window) < N: # 未满
                if vtype == '0':
                    # 判断是否在通信范围内
                    last_cav_id = None
                    for id in current_window:
                        if id[2] == '0':
                            last_cav_id = id
                    if last_cav_id == None: # 窗口内无CAV
                        current_window.append(vid)
                    else:
                        last_pos = traci.vehicle.getPosition(last_cav_id)[0]
                        if abs(last_pos - pos) < R:
                            current_window.append(vid)
                        else:
                            windows.append(current_window)
                            current_window = []
                            current_window.append(vid)
                else:
                    current_window.append(vid)
            if len(current_window) == N:
                windows.append(current_window)
                current_window = []
        if current_window:
            windows.append(current_window)

        # clear_window


        return windows

    def sliding_window_v2(self, N, R):
        """
        滑动窗口算法
        N -> max num of vehicels within a window
        R -> max CAV communication distance
        动态窗口, 截取已经进入E2且在瓶颈区域前所有车辆划分窗口
        """
        windows = []
        current_window = []
        # vids = self.get_total_vids()
        vids = self.get_not_window_vids()

        # 从前往后访问
        for vid in vids[::-1]:
            vtype = vid[2] # 'v.0.n' -> CAV
            pos = traci.vehicle.getPosition(vid)[0]
            # if pos > self.accident_pos or pos < 1000:
            #     continue
            if pos < 1200:
                continue
            if not current_window:  # 为空
                current_window.append(vid)
            elif len(current_window) < N: # 未满
                if vtype == '0':
                    # 判断是否在通信范围内
                    last_cav_id = None
                    for id in current_window:
                        if id[2] == '0':
                            last_cav_id = id
                    if last_cav_id == None: # 窗口内无CAV
                        current_window.append(vid)
                    else:
                        last_pos = traci.vehicle.getPosition(last_cav_id)[0]
                        if abs(last_pos - pos) < R:
                            current_window.append(vid)
                        else:
                            windows.append(current_window)
                            current_window = []
                            current_window.append(vid)
                else:
                    current_window.append(vid)
            if len(current_window) == N:
                windows.append(current_window)
                current_window = []
        if current_window:
            windows.append(current_window)

        return windows

    def perform_action(self, agent_vids, action, leader_id):
        """
        执行动作, 先变道, 再追车
        因为是非瞬时动作, 需要将变道以及追车信息都导入change_lane_state和join_state里面
        在之后检查动作的完成度
        """
        # 将智能体执行动作的次序排好, 当同时执行动作时, 位置更靠前的智能体先执行动作
        action_list = []
        for i in range(len(agent_vids)):
            vid = agent_vids[i]
            pos = traci.vehicle.getPosition(vid)
            a = action[i]
            action_list.append([vid, a, pos[0]])
        # print(action_list)
        action_list = sorted(action_list, key=lambda x:x[-1], reverse=True)
        # print('action_list:', action_list)

        # change lane frist
        for a in action_list:
            if a[1] == 1:
                # perform action
                vid = a[0]
                traci.vehicle.setLaneChangeMode(vid, 512)
                lane = traci.vehicle.getLaneIndex(leader_id)
                if vid not in self.change_lane_state:
                    self.change_lane_state[vid] = {'state': 0, 'lane': lane}
                # traci.vehicle.changeLane(vid, lane, duration=5)

                if vid not in self.join_state:
                    self.join_state[vid] = {'state' : -1, 'front' : leader_id, 'leader' : leader_id}

        action_list.clear()

    def is_vid_in_join_state(self, vid):
        "判断vid是否在join_state里面，也就是是否执行了动作"
        # 遍历join_state中的每个键值对
        for key, value in self.join_state.items():
            if vid == key:
                return True
        return False

    def Is_change_lane(self):
        """
        检测变道是否完成, 必须先完成变道再执行之后的动作
        完成变道之后, 再允许车辆去追逐加入车队
        """
        if self.change_lane_state:
            for i in self.change_lane_state.keys():
                state = self.change_lane_state[i]['state']
                lane = self.change_lane_state[i]['lane']
                if state == 0:
                    # print(i, '未完成变道, 执行变道动作')
                    traci.vehicle.setLaneChangeMode(i, 512)
                    traci.vehicle.changeLane(i, lane, duration=5)
                    if traci.vehicle.getLaneIndex(i) == lane:
                        self.change_lane_state[i]['state'] = 1
                        print(i, '已完成变道')
                        traci.vehicle.setLaneChangeMode(i, 0)
                        self.join_state[i]['state'] = GOING_TO_POSITION
                        break

    def Is_joined_platoon(self):
        """
        检测执行动作的智能体CAV是否可以开始追击以及是否完成追击操作,
        当距离合适时, 提示完成合并
        """
        if self.join_state:
            for vid in self.join_state.keys():
                lane = traci.vehicle.getLaneIndex(vid)
                if lane != self.platoon_lane:
                    continue
                v_state = self.join_state[vid]['state']
                leader_id = self.join_state[vid]['leader']
                if v_state == GOING_TO_POSITION:
                    fid = self.get_front_lane_vid(vid)
                    self.join_state[vid]['front'] = fid

                    leader_id = self.get_front_leader_vid(vid)
                    if leader_id == None:
                        continue
                    self.join_state[vid]['leader'] = leader_id

                    traci.vehicle.setSpeedMode(vid, 0)
                    # traci.vehicle.setMaxSpeed(vid, MAX_SPEED)
                    # 加入拓扑结构
                    self.topology[vid] = {"leader": leader_id, "front": fid}
                    self.plexe.set_cc_desired_speed(vid, MAX_SPEED + 3)
                    self.plexe.set_active_controller(vid, FAKED_CACC)

                    if get_distance(self.plexe, vid, fid) < DISTANCE * 2:
                        print('距离够了完成合并')
                        self.plexe.set_path_cacc_parameters(vid, distance=5)
                        self.plexe.set_active_controller(vid, CACC)
                        self.join_state[vid]['state'] = COMPLETED
                        # 检测leader是否需要恢复速度
                        # if self.all_followers_joined(self.topology[i]['leader']):
                        #     self.plexe.set_cc_desired_speed(self.topology[i]['leader'], SPEED)
                        break

    def Is_leader_get_platoon_lane(self, leader_id):
        "检测leader是否以及到达了专用车道"
        lane_index = traci.vehicle.getLaneIndex(leader_id)
        if lane_index == self.platoon_lane:
            return True
        else:
            self.plexe.set_fixed_lane(leader_id, self.platoon_lane, True)
            # traci.vehicle.setLaneChangeMode(leader_id, 512)
            # traci.vehicle.changeLane(leader_id, self.platoon_lane, duration=0)
            return False

    def Is_done(self, window):
        '''
        该窗口是否完成 done
        1 -> 发生碰撞
        2 -> 最后一辆CAV通过瓶颈区域
        '''
        collision = [element for sublist in self.collision_vids for element in sublist]
        # 求交集
        intersection = list(set(collision).intersection(set(window)))
        # print('intersection:', intersection)
        if len(intersection) != 0:
            print('该窗口内已经发生碰撞')
            self.done = True
            return

        cav_ids = [item for item in window if item.startswith('v.0.')]
        print('cav_ids:', cav_ids)
        last_vid, last_pos = self.get_last_vid(cav_ids)
        print('last_vid:', last_vid, 'last_pos:', last_pos, 'accident_pos:', self.accident_pos)
        if last_pos > self.accident_pos:
            self.done = True

    def Is_window_done(self, window):
        "判断该窗口是否已经执行过动作, 执行过后就不再执行算法"
        cav_vids = self.get_window_cav_vids(window)

        for vid in cav_vids:
            if vid not in self.join_state:
                return False

        return True

    def Is_pass_bottleneck(self, window):
        '''
        该窗口内最后一辆CAV是否已经通过瓶颈
        '''
        cav_ids = [item for item in window if item.startswith('v.0.')]
        print('cav_ids:', cav_ids)
        last_vid, last_pos = self.get_last_vid(cav_ids)
        print('last_vid:', last_vid, 'last_pos:', last_pos, 'accident_pos:', self.accident_pos)
        if last_pos > self.accident_pos:
            self.done = True

    def marked_action(self, action, agent_ids):
        '''
        检查CAV的动作状态, 看是否需要对动作施加限制
        对于已经开始执行合并动作的CAV, 对其后的动作进行限制
        '''
        # print(self.join_state)
        for key, item in self.join_state.items():
            state_value = item['state']
            if state_value == GOING_TO_POSITION or state_value == -1:
                # self.Is_joined_platoon()
                vid = key
                for i in range(len(agent_ids)):
                    if agent_ids[i] == vid:
                        action[i] = 0
            elif state_value == COMPLETED:
                vid = key
                for i in range(len(agent_ids)):
                    if agent_ids[i] == vid:
                        action[i] = 0
        return action

    def decimal_to_binary_list(self, decimal_num, num_bits):
        '''
        将十进制代表的动作转化为二进制的数组,
        例如 0 -> [0, 0]; 1 -> [0, 1]
        :param decimal_num: 十进制
        :param num_bits: 转化的位数, 即列表长度或智能体数量
        :return: 列表, action
        '''
        binary_str = format(decimal_num, f'0{num_bits}b')
        binary_list = [int(bit) for bit in binary_str]
        return binary_list

    def process_data(self):
        """
        加载数据, 用于训练
        """
        data = random.sample(self.memory, self.batch)
        s = np.array([d[0] for d in data])
        a = [d[1] for d in data]
        s_ = np.array([d[2] for d in data])
        r = [d[3] for d in data]
        done = [d[4] for d in data]

        y = self.Q_network(np.array(s, dtype='float32'))
        y = y.numpy()
        Q1 = self.target_Q_network(np.array(s_, dtype='float32'))
        Q2 = self.Q_network(np.array(s_, dtype='float32'))
        next_action = np.argmax(Q2, axis=1)

        for i, (_, a, _, r, done) in enumerate(data):
            if done:
                target = r
            else:
                target = r + self.gamma * Q1[i][next_action[i]]
            target = np.array(target, dtype='float32')

            y[i][a] = target

        return s, y

    def update_Q_network(self, s, y):
        """
        更新Q网络, 返回损失loss
        """
        with tf.GradientTape() as tape:
            Q = self.Q_network(np.array(s, dtype='float32'))
            loss = tl.cost.mean_squared_error(Q, y)
        grads = tape.gradient(loss, self.Q_network.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.Q_network.trainable_weights))

        return loss

    def update_target_Q(self):
        """
        更新目标网络
        """
        for i, target in zip(self.Q_network.trainable_weights, self.target_Q_network.trainable_weights):
            target.assign(i)

    def train(self):
        '''
        神经网络的训练
        '''
        s, y = self.process_data()
        loss = self.update_Q_network(s, y)
        self.total_loss.append(loss)
        if (self.episode + 1) % 100 == 0:
            self.update_target_Q()

    def normalize_speed(self, speed):
        '''
        将速度标准化映射为[-1, 1]
        '''
        desire_speed = 30.55
        only_leader_speed = 31.48
        max_speed = 33.33
        speed_normaliazed = (speed - only_leader_speed) / (max_speed - only_leader_speed)

        return speed_normaliazed

    def remember(self, s, a, s_, r, done):
        """
        remember the data to experience replay
        """
        data = (s, a, s_, r, done)
        # print('data:', data)
        self.memory.append(data)
        # memory_str = '\n'.join(str(item) for item in self.memory)
        # out_file = 'output/output.txt'
        # with open(out_file, 'w') as file:
        #     file.write(memory_str)
        return self.memory

    def start_CAV_platoon(self, window):
        """
        判断是否开启车辆编队算法
        1) 窗口内的所有车都在编队区域
        2) 车辆速度不能太低 -> 出VSL后车间距不会太小
        """
        max_speed = 0
        for vid in window:
            lane = traci.vehicle.getLaneID(vid)[:2]
            speed = round(traci.vehicle.getSpeed(vid), 2)
            if speed > max_speed:
                max_speed = speed
            if lane != self.start_platoon_lane:
                return False

        if max_speed > 29:
            return True
        else:
            return False

    def start_sumo(self):
        "只测试环境, 什么也不做"

        start_sumo("cfg/freeway.sumo.cfg", False, gui=self.rend)
        self.plexe = Plexe()
        traci.addStepListener(self.plexe)

        while running(True, self.step, self.simulation_step):
            if self.step == self.simulation_step:
                self.episode = self.episode + 1
                if self.episode >= self.episode_max:
                    print('仿真结束')
                    break
                self.reinit()
                start_sumo("cfg/freeway.sumo.cfg", True)

            traci.simulationStep()

            if self.step ==  0:
                # self.create_vehicles()
                self.create_vehicles_new_v2()
            # if self.step % 2000 == 0 or self.step == 0:
            #     self.create_vehciles_new(v_num=30, CAV_penetration_rate=0.6)

            if self.step == 10 and self.rend:
                random_vid = traci.vehicle.getIDList()[0]
                traci.gui.trackVehicle("View #0", random_vid)
                traci.gui.setZoom("View #0", 1500)

            if self.step == 200:
                self.create_accident()

            # if self.step % 50 == 0:
            #     self.vsl()

            self.step += 1

        traci.close()

    def start_small_sumo(self):

        start_sumo("cfg/freeway.sumo.cfg", False, gui=self.rend)
        self.plexe = Plexe()
        traci.addStepListener(self.plexe)
        self.simulation_step = 20000
        while running(True, self.step, self.simulation_step):
            if self.step == self.simulation_step:
                self.episode = self.episode + 1
                if self.episode >= self.episode_max:
                    print('仿真结束')
                    break
                self.reinit()
                start_sumo("cfg/freeway.sumo.cfg", True)

            traci.simulationStep()

            # if self.step ==  0:
            #     # self.create_vehicles()
            #     self.create_vehicles_new_v2()
            # if self.step % 2000 == 0 or self.step == 0:
            #     self.create_vehciles_new(v_num=30, CAV_penetration_rate=0.6)

            if self.step == 10 and self.rend:
                random_vid = traci.vehicle.getIDList()[0]
                traci.gui.trackVehicle("View #0", random_vid)
                traci.gui.setZoom("View #0", 1500)

            if self.step == 200:
                self.create_accident()

            if self.step % 50 == 0:
                self.vsl()

            self.step += 1

        traci.close()

    def start_large_sumo(self):
        "大规模测试"

        start_sumo("cfg/freeway.sumo.cfg", False, gui=self.rend)
        self.plexe = Plexe()
        traci.addStepListener(self.plexe)

        while running(True, self.step, self.simulation_step):
            if self.step == self.simulation_step:
                self.episode = self.episode + 1
                # 记录数据
                print('episode: ', self.episode, '; total_reward: ', self.total_reward)
                print('------第', self.episode, '回合仿真结束------')
                with open(self.file_path + '_large_test_total_reward.txt', 'a') as file:
                    file.write(f"Episode {self.episode}: Total Reward = {self.total_reward}\n")

                print('join_state:', self.join_state)

                if self.episode >= self.episode_max:
                    print('-----测试结束-----')
                    break

                self.reinit()

                # 重新开启下回合的训练
                start_sumo("cfg/freeway.sumo.cfg", True, gui=self.rend)

            traci.simulationStep()

            colliding_vehicles = list(traci.simulation.getCollidingVehiclesIDList())
            if len(colliding_vehicles) != 0:
                print('发生碰撞,车辆ID为:', colliding_vehicles)
                self.collision_vids.append(colliding_vehicles)
                self.collision = True

            if self.step  % 3000 ==  0:
                # self.create_vehciles_new(v_num=30, CAV_penetration_rate=0.6)
                self.create_vehicles_new_v2(v_num=30, p=0.6)

            # if self.step % 2000 ==  0:
            #     self.create_vehicles()
                # self.create_vehicles_random()

            if self.step == 10 and self.rend:
                random_vid = traci.vehicle.getIDList()[0]
                traci.gui.trackVehicle("View #0", random_vid)
                traci.gui.setZoom("View #0", 1500)


            if self.step == 200:
                self.create_accident()

            if self.step > 2000 and self.step % 200 == 0:
                windows = self.sliding_windows_new(N = 5, R = self.R)
                print('windows:', windows)
                for window in windows:
                    # 对于 0 or 1 辆CAV的窗口需要特殊处理
                    print('for window:', window)
                    # 对于已经执行过的窗口应该直接跳过算法
                    if self.Is_window_done(window):
                        print('该窗口已经执行完成')
                        continue

                    cav_vids_within_window = [item for item in window if item.startswith('v.0.')]

                    if len(cav_vids_within_window) == 0:
                        print('该窗口内无CAV, 不执行车辆编队')
                        continue

                    if len(cav_vids_within_window) == 1 and cav_vids_within_window[0] not in self.leader_vids \
                            and cav_vids_within_window[0] not in self.join_state and not self.done \
                            and traci.vehicle.getLaneID(cav_vids_within_window[0])[:2] == 'E2' \
                            and self.start_CAV_platoon(window):
                        print('该窗口内只有一辆CAV:', cav_vids_within_window[0], ', 需要判断是否要加入前方窗口的车队')
                        front_vid = self.get_front_vid(cav_vids_within_window[0])
                        if traci.vehicle.getPosition(front_vid)[0] - \
                                traci.vehicle.getPosition(cav_vids_within_window[0])[0] < self.R:
                            self.perform_action(cav_vids_within_window, [1], leader_id=self.leader_vids[-1])
                        else:
                            print('通信距离不够, 不加入前方车队')
                            continue

                    # 判断是否要开始执行编队算法
                    if self.start_CAV_platoon(window) and not self.done:
                        print('执行车辆编队算法')
                        self.select_platoon_lane()
                        leader = self.select_leader(window)
                        if leader == 'v.0.0':
                            break
                        if not self.Is_leader_get_platoon_lane(leader):
                            break
                        vids = window[::]
                        vids.remove(leader)
                        cav_ids = self.get_window_cav_vids(window)

                        # get state
                        state = self.get_state(vids, leader)
                        print('state:', state)
                        # print('state_len:', len(state))
                        # get action
                        action = []
                        if self.episode < 300:
                            for i in range(len(cav_ids)):
                                action.append(1)
                        else:
                            action = self.get_action(state, cav_ids)

                        # marked action
                        marked_action = self.marked_action(action, cav_ids)

                        # perfrom action
                        print('leader:', leader, 'cav_ids:', cav_ids, 'action:', marked_action)
                        self.perform_action(cav_ids, marked_action, leader)

            if self.step % 50 == 0:
                self.vsl()
                self.Is_joined_platoon()
                self.Is_change_lane()
                for leader_vid in self.leader_vids:
                    self.Is_leader_get_platoon_lane(leader_vid)
                if self.platoon_lane >= 0:
                    self.clear_platoon_lane()
                # print('join_state:', self.join_state)
                communicate(self.plexe, self.topology)
                # print('change_lane_state:', self.change_lane_state)
                # print('collision_vids:', self.collision_vids)

            self.step += 1

        traci.close()

if __name__ == "__main__":
    Freeway = Freeway()
    # Freeway.start_large_sumo()
    Freeway.start_small_sumo()
