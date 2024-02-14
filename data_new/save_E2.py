#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：train_freeway.sumo.cfg 
@File    ：save_E2.py
@Author  ：Lu
@Date    ：2023/11/10 18:59 
'''
import xml.etree.ElementTree as ET
from openpyxl import Workbook

# 解析XML文件
tree = ET.parse('output.xml')
root = tree.getroot()

# 创建Excel工作簿和工作表
workbook = Workbook()
worksheet = workbook.active

# 添加表头
headers = ["Interval Begin", "Interval End", "ID", "Sampled Seconds", "nVehEntered", "nVehLeft", "nVehSeen",
           "Mean Speed", "Mean Time Loss", "Mean Occupancy", "Max Occupancy", "Mean Max Jam Length in Vehicles",
           "Mean Max Jam Length in Meters", "Max Jam Length in Vehicles", "Max Jam Length in Meters",
           "Jam Length in Vehicles Sum", "Jam Length in Meters Sum", "Mean Halting Duration", "Max Halting Duration",
           "Halting Duration Sum", "Mean Interval Halting Duration", "Max Interval Halting Duration",
           "Interval Halting Duration Sum", "Started Halts", "Mean Vehicle Number", "Max Vehicle Number"]
worksheet.append(headers)

# 遍历XML数据并写入Excel文件
for interval in root.findall('.//interval'):
    data = [
        interval.get('begin'),
        interval.get('end'),
        interval.get('id'),
        interval.get('sampledSeconds'),
        interval.get('nVehEntered'),
        interval.get('nVehLeft'),
        interval.get('nVehSeen'),
        interval.get('meanSpeed'),
        interval.get('meanTimeLoss'),
        interval.get('meanOccupancy'),
        interval.get('maxOccupancy'),
        interval.get('meanMaxJamLengthInVehicles'),
        interval.get('meanMaxJamLengthInMeters'),
        interval.get('maxJamLengthInVehicles'),
        interval.get('maxJamLengthInMeters'),
        interval.get('jamLengthInVehiclesSum'),
        interval.get('jamLengthInMetersSum'),
        interval.get('meanHaltingDuration'),
        interval.get('maxHaltingDuration'),
        interval.get('haltingDurationSum'),
        interval.get('meanIntervalHaltingDuration'),
        interval.get('maxIntervalHaltingDuration'),
        interval.get('intervalHaltingDurationSum'),
        interval.get('startedHalts'),
        interval.get('meanVehicleNumber'),
        interval.get('maxVehicleNumber')
    ]
    worksheet.append(data)

# 保存Excel文件
workbook.save('test6_E2_output.xlsx')
