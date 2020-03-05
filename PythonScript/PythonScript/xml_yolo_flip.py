# -*- coding: utf-8 -*-
#此代码可以将xml文件格式转换为yolo格式  !!二次执行需删除前一次txt文件
import os
import glob
import shutil
import xml.etree.ElementTree as ET
label_dict = {'Helicopter':0,'Person':1,'LargeVehicle' : 2,'LightVehicle':3,'HeavyVehicle' : 4,}
dirpath =  r'F:\VIDEOS\20200217坦克标注\20200117\xml/'   # 原来存放xml文件的目录
newdir = r'F:\VIDEOS\20200217坦克标注\20200117\pict_flip/'   # 修改label后形成的txt目录
errorpath = r'F:\error/'

if not os.path.exists(newdir):
    os.makedirs(newdir)
if not os.path.exists(errorpath):
    os.makedirs(errorpath)
os.chdir(dirpath)
for fp in glob.glob('*.xml'):
    error_name = '.'.join(fp.split('.')[:-1])+ '.xml'
    error_jname = '.'.join(fp.split('.')[:-1])+ '.jpg'
    label_name = '.'.join(fp.split('.')[:-1])+ '.txt'
    #
    jpg_path = dirpath + error_jname
    error_jpath = errorpath + error_jname
    txt_path = dirpath + error_name
    error_path = errorpath + error_name

    # try:
    root = ET.parse(os.path.join(dirpath, fp)).getroot()
    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    sz = root.find('size')
    width = float(sz.find('width').text)
    height = float(sz.find('height').text)
    filename = root.find('filename').text
    try:
        for child in root.findall('object'):  # 找到图片中的所有框
            flag = 0
            sub = child.find('bndbox')  # 找到框的标注值并进行读
            print(child.find('name').text)
            label = label_dict[child.find('name').text]
            xmin = float(sub.find('xmin').text)
            ymin = float(sub.find('ymin').text)
            xmax = float(sub.find('xmax').text)
            ymax = float(sub.find('ymax').text)

            xmin_flip = float(width - xmax)
            xmax_flip = float(width - xmin)


            x_center = round((xmin_flip + xmax_flip) / (2 * width), 6)
            y_center = round((ymin + ymax) / (2 * height), 6)
            w = round((xmax_flip - xmin_flip) / width, 6)
            h = round((ymax - ymin) / height, 6)
            with open(os.path.join(newdir,label_name), 'a+') as f:
                f.write(' '.join([str(label), str(x_center), str(y_center), str(w), str(h) + '\n']))
    except ZeroDivisionError:
        shutil.move(txt_path, error_path)
        shutil.move(jpg_path, error_jpath)
print('ok')