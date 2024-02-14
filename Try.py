#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：train_freeway.sumo.cfg 
@File    ：Try.py
@Author  ：Lu
@Date    ：2023/11/5 18:22 
'''
import queue
import random
#
# def create_vehicles_random(self):
#     """
#     带有随机性创建车辆, 位置车道固定, 但车辆类型不确定
#     num = 15, CAV占60%
#     """
#     max_cav_value, max_hdv_value = self.get_max_vid_value()
#     if max_cav_value == -1:
#         max_cav_value = 0
#     if max_hdv_value == -1:
#         max_hdv_value = 0
#     # print(max_cav_value, max_hdv_value)
#     # position = [[3, 4, 10], [30, 33, 40], [53, 60, 64], [85, 88, 100], [105, 100, 125]]
#     # lane_index = [[2, 1, 0], [0, 2, 1], [2, 1, 0], [2, 1, 0], [2, 0, 1]]
#     position = [[3, 4, 8], [33, 34, 40], [63, 66, 74], [95, 107, 100], [140, 130, 135]]
#     lane_index = [[2, 1, 0], [0, 2, 1], [2, 1, 0], [2, 1, 0], [2, 0, 1]]
#     # vtype = [[1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0]]
#     # vtype = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 1]]
#     vtype = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0]]
#     vtype = self.get_random_matrix(5, 3)
#     #
#     print(vtype)
#     for i in range(len(position)):
#         for j in range(len(position[i])):
#             pos = position[i][j]
#             index = lane_index[i][j]
#             type = vtype[i][j]
#             if type == 0:
#                 vid = 'v.0.' + str(1 + max_cav_value)
#                 max_cav_value += 1
#                 traci.vehicle.add(vehID=vid, routeID="highway_route", typeID="vtypeauto",
#                                   departLane=index, departPos=pos, departSpeed=20.00)
#             elif type == 1:
#                 vid = 'v.1.' + str(1 + max_hdv_value)
#                 max_hdv_value += 1
#                 traci.vehicle.add(vehID=vid, routeID="highway_route", typeID="passenger",
#                                   departLane=index, departPos=pos, departSpeed=20.00)


import xml.etree.ElementTree as ET
from openpyxl import Workbook

# 解析XML文件
tree = ET.parse('output.xml')
root = tree.getroot()

# 创建Excel工作簿和工作表
workbook = Workbook()
worksheet = workbook.active

# 添加表头
headers = ["Interval Begin", "Interval End", "ID", "nVehContrib", "Flow", "Occupancy", "Speed", "HarmonicMeanSpeed", "Length", "nVehEntered"]
worksheet.append(headers)

# 遍历XML数据并写入Excel文件
for interval in root.findall('.//interval'):
    data = [
        interval.get('begin'),
        interval.get('end'),
        interval.get('id'),
        interval.get('nVehContrib'),
        interval.get('flow'),
        interval.get('occupancy'),
        interval.get('speed'),
        interval.get('harmonicMeanSpeed'),
        interval.get('length'),
        interval.get('nVehEntered')
    ]
    worksheet.append(data)

# 保存Excel文件
workbook.save('E1_output_test1.xlsx')






