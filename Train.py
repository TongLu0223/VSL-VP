#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：FREEWAY
@File    ：Train.py
@Author  ：Lu
@Date    ：2023/11/3 10:58 
'''
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
from FREEWAY import Freeway
from plexe import ACC, CACC, FAKED_CACC
from datetime import datetime
from ccparams import *
from utils import *


class Train(Freeway.Freeway):

    def __init__(self):

        super().__init__()
        self.rend = True
        self.save_model_interval = 1000
        self.start_train = 200
        self.episode_max = 1
        self.simulation_step = 48000

        self.sumo_cfg = "cfg/train_freeway.sumo.cfg"

    def train_sumo(self):
        "start training for five vehicles -> 3 CAVs and 2 HDVs"

        start_sumo(self.sumo_cfg, False, gui=self.rend)
        self.plexe = Plexe()
        traci.addStepListener(self.plexe)

        while running(True, self.step, self.simulation_step):
            if self.step == self.simulation_step:
                self.episode += 1
                print('episode: ', self.episode, '; total_reward: ', self.total_reward)
                print('join_state:', self.join_state)
                print('------第', self.episode, '回合仿真结束------')
                with open(self.file_path + '_total_reward.txt', 'a') as file:
                    file.write(f"Episode {self.episode}: Total Reward = {self.total_reward}\n")

                if (self.episode + 1) % self.save_model_interval == 0 or self.episode == 1:
                    print('保存模型')
                    self.Q_network.save(filepath= self.file_path +'_' + str(self.episode) + '_model.h5', save_weights = True)

                if self.episode >= self.episode_max:
                    print('-----训练结束-----')
                    break

                self.reinit()

                if (self.episode + 1) % 10 == 0 and self.episode > 300:
                    self.update_epsilon()
                    print('epsiode:',self.episode,'epsilon:', self.epsilon)

                # 重新开启下回合的训练
                start_sumo(self.sumo_cfg, True, gui=self.rend)

            traci.simulationStep()

            colliding_vehicles = list(traci.simulation.getCollidingVehiclesIDList())
            if len(colliding_vehicles) != 0:
                print('发生碰撞,车辆ID为:', colliding_vehicles)
                self.collision_vids.append(colliding_vehicles)
                self.collision = True

            if self.step == 0 and self.rend:
                random_vid = traci.vehicle.getIDList()[0]
                traci.gui.trackVehicle("View #0", random_vid)
                traci.gui.setZoom("View #0", 1500)

            if self.step == 200:
                self.create_accident()


            if self.step > 2000 and self.step % 200 == 0:
                windows = self.sliding_windows(N = 5, R = 1000)
                for window in windows:
                    print('for window:', window)
                    cav_vids_within_window = [item for item in window if item.startswith('v.0.')]

                    if len(cav_vids_within_window) == 0:
                        print('该窗口内无CAV, 不执行车辆编队')
                        continue

                    if len(cav_vids_within_window) == 1 and \
                        cav_vids_within_window[0] not in self.leader_vids and \
                        cav_vids_within_window[0] not in self.join_state and not self.done :
                        print('该窗口内只有一辆CAV:', cav_vids_within_window[0], ', 需要判断是否要加入前方窗口的车队')
                        front_vid = self.get_front_vid(cav_vids_within_window[0])
                        if traci.vehicle.getPosition(front_vid)[0] - \
                                traci.vehicle.getPosition(cav_vids_within_window[0])[0] < self.R:
                                if traci.vehicle.getLaneID(cav_vids_within_window[0])[:2] == 'E2':
                                    self.perform_action(cav_vids_within_window, [1], leader_id=self.leader_vids[-1])
                        else:
                            print('通信距离不够, 不加入前方车队')
                            continue

                    if self.start_CAV_platoon(window) and not self.done:
                        print('Vehicle Platooning Start')
                        self.select_platoon_lane()
                        leader = self.select_leader(window)
                        if leader == 'v.0.0':
                            break
                        if not self.Is_leader_get_platoon_lane(leader):
                            break
                        vids = window[::]
                        vids.remove(leader)
                        cav_ids = self.get_window_cav_vids(window)

                        # remember data (s,a,s',r,done)
                        if self.first_action:
                            self.first_action = False
                        else:
                            next_state = self.get_state(vids, leader)
                            self.Is_done(window)
                            reward = self.get_reward_shaping(cav_ids, leader)
                            print('reward:', reward)
                            self.total_reward += reward
                            self.memory = self.remember(state, action, next_state, reward, self.done)
                            print('done:', self.done)
                            if self.done:
                                continue

                        # get state
                        state = self.get_state(vids, leader)
                        print('state:', state)

                        # get action
                        action = []
                        if self.episode < 100:
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
                if self.platoon_lane >= 0:
                    self.clear_platoon_lane()
                communicate(self.plexe, self.topology)
                # print('join_state:', self.join_state)
                # print('collision_vids:', self.collision_vids)

            # train
            if self.episode > self.start_train and len(self.memory) > self.batch \
                    and self.step % 1000 == 0:
                self.train()

            self.step += 1

        traci.close()

    def test_sumo(self):
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

            if self.step % 2000 ==  0:
                # self.create_vehicles()
                self.create_vehicles_new_v2(v_num=30, p=0.6)
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

    def test_large_sumo(self):
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

            if self.step % 1500 ==  0:
                # self.create_vehciles_new(v_num=30, CAV_penetration_rate=0.6)
                self.create_vehicles_new_v2(v_num=30, p=0.6)

            # if self.step % 2000 ==  0:
            #     self.create_vehicles()
                # self.create_vehicles_random()

            if self.step == 10 and self.rend:
                random_vid = traci.vehicle.getIDList()[0]
                traci.gui.trackVehicle("View #0", random_vid)
                traci.gui.setZoom("View #0", 7000)


            if self.step == 200:
                self.create_accident()

            if self.step > 2000 and self.step % 200 == 0:
                while not self.windows.empty():
                    window = self.windows.get()
                    print(window)

            if self.step > 2000 and self.step % 500 == 0:
                windows = self.sliding_window_v2(N = 5, R = self.R)
                print('windows:', windows)
                for window in windows:
                    # 对于 0 or 1 辆CAV的窗口需要特殊处理
                    if len(window) != 5:
                        continue
                    print('for window:', window)

                    cav_vids_within_window = [item for item in window if item.startswith('v.0.')]

                    if len(cav_vids_within_window) == 0:
                        print('该窗口内无CAV, 不执行车辆编队')
                        continue

                    # if len(cav_vids_within_window) == 1 and cav_vids_within_window[0] not in self.leader_vids \
                    #         and cav_vids_within_window[0] not in self.join_state and not self.done \
                    #         and traci.vehicle.getLaneID(cav_vids_within_window[0])[:2] == 'E2' \
                    #         and self.start_CAV_platoon(window):
                    #     print('该窗口内只有一辆CAV:', cav_vids_within_window[0], ', 需要判断是否要加入前方窗口的车队')
                    #     front_vid = self.get_front_vid(cav_vids_within_window[0])
                    #     if traci.vehicle.getPosition(front_vid)[0] - \
                    #             traci.vehicle.getPosition(cav_vids_within_window[0])[0] < self.R:
                    #         self.perform_action(cav_vids_within_window, [1], leader_id=self.leader_vids[-1])
                    #     else:
                    #         print('通信距离不够, 不加入前方车队')
                    #         continue

                    # 判断是否要开始执行编队算法
                    if self.start_CAV_platoon(window) and not self.done:
                        print('执行车辆编队算法')
                        self.select_platoon_lane()
                        leader = self.select_leader(window)
                        # for vid in window:
                        #     self.window_done_vids.append(vid)
                        if leader == 'v.0.0':
                            continue
                        if not self.Is_leader_get_platoon_lane(leader):
                            continue
                        vids = window[::]
                        vids.remove(leader)
                        cav_ids = self.get_window_cav_vids(window)

                        # get state
                        state = self.get_state(vids, leader)
                        print('state:', state)
                        # print('state_len:', len(state))
                        # get action
                        action = []
                        if self.episode < 30:
                            for i in range(len(cav_ids)):
                                action.append(1)
                        else:
                            action = self.get_action(state, cav_ids)

                        # marked action
                        marked_action = self.marked_action(action, cav_ids)

                        # perfrom action
                        print('leader:', leader, 'cav_ids:', cav_ids, 'action:', marked_action)
                        self.perform_action(cav_ids, marked_action, leader)

                    # # 对于已经执行过的窗口应该直接跳过算法
                    if self.Is_window_done(window):
                        print('该窗口已经执行完成')
                        for vid in window:
                            self.window_done_vids.append(vid)
                        continue
            #
            if self.step % 50 == 0:
                self.vsl()
                self.Is_joined_platoon()
                self.Is_change_lane()
                for leader_vid in self.leader_vids:
                    self.Is_leader_get_platoon_lane(leader_vid)
                # print('join_state:', self.join_state)

                # print('change_lane_state:', self.change_lane_state)
                # print('collision_vids:', self.collision_vids)
            if self.step % 10 == 0:
                communicate(self.plexe, self.topology)

            if self.step % 500 == 0:
                if self.platoon_lane >= 0:
                    self.clear_platoon_lane()

            self.step += 1

        traci.close()

train = Train()
# train.train_sumo()
# train.test_sumo()
train.test_large_sumo()
