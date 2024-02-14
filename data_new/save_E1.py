#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：train_freeway.sumo.cfg 
@File    ：save_E1.py
@Author  ：Lu
@Date    ：2023/11/10 19:04 
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
workbook.save('test6_E1_output.xlsx')