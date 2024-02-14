#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：train_freeway.sumo.cfg 
@File    ：save_E3.py
@Author  ：Lu
@Date    ：2023/11/10 19:01 
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
headers = ["Interval Begin", "Interval End", "ID", "Mean Travel Time", "Mean Overlap Travel Time", "Mean Speed",
           "Mean Halts Per Vehicle", "Mean Time Loss", "Vehicle Sum", "Mean Speed Within",
           "Mean Halts Per Vehicle Within", "Mean Duration Within", "Vehicle Sum Within",
           "Mean Interval Speed Within", "Mean Interval Halts Per Vehicle Within", "Mean Interval Duration Within",
           "Mean Time Loss Within"]
worksheet.append(headers)

# 遍历XML数据并写入Excel文件
for interval in root.findall('.//interval'):
    data = [
        interval.get('begin'),
        interval.get('end'),
        interval.get('id'),
        interval.get('meanTravelTime'),
        interval.get('meanOverlapTravelTime'),
        interval.get('meanSpeed'),
        interval.get('meanHaltsPerVehicle'),
        interval.get('meanTimeLoss'),
        interval.get('vehicleSum'),
        interval.get('meanSpeedWithin'),
        interval.get('meanHaltsPerVehicleWithin'),
        interval.get('meanDurationWithin'),
        interval.get('vehicleSumWithin'),
        interval.get('meanIntervalSpeedWithin'),
        interval.get('meanIntervalHaltsPerVehicleWithin'),
        interval.get('meanIntervalDurationWithin'),
        interval.get('meanTimeLossWithin')
    ]
    worksheet.append(data)

# 保存Excel文件
workbook.save('test6_E3_output.xlsx')
