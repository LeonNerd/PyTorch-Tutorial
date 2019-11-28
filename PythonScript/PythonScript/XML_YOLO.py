# -*- coding: utf-8 -*-
#此代码可以将xml文件格式转换为yolo格式  !!二次执行需删除前一次txt文件
import os
import glob
import shutil
import xml.etree.ElementTree as ET
label_dict = {'Helicopter':0,'Person':1 }
dirpath =  'H:/data/微光相机视频/20191128微光相机标注/xml/'  # 原来存放xml文件的目录
newdir = 'H:/data/微光相机视频/20191128微光相机标注/yolo/'  # 修改label后形成的txt目录
errorpath = "H:/data/微光相机视频/20191128微光相机标注/error/"

if not os.path.exists(newdir):
    os.makedirs(newdir)
if not os.path.exists(errorpath):
    os.makedirs(errorpath)
os.chdir(dirpath)
for fp in glob.glob('*.xml'):
    error_name = '.'.join(fp.split('.')[:-1])+ '.xml'
    print(error_name)
    label_name = '.'.join(fp.split('.')[:-1])+ '.txt'

    txt_path = dirpath + error_name
    error_path = errorpath + error_name

    # try:
    root = ET.parse(os.path.join(dirpath, fp)).getroot()
    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    sz = root.find('size')
    width = float(sz.find('width').text)
    height = float(sz.find('height').text)
    filename = root.find('filename').text

    for child in root.findall('object'):  # 找到图片中的所有框
        flag = 0
        sub = child.find('bndbox')  # 找到框的标注值并进行读
        print(child.find('name').text)
        label = label_dict[child.find('name').text]
        xmin = float(sub.find('xmin').text)
        ymin = float(sub.find('ymin').text)
        xmax = float(sub.find('xmax').text)
        ymax = float(sub.find('ymax').text)
        x_center = round((xmin + xmax) / (2 * width), 6)
        y_center = round((ymin + ymax) / (2 * height), 6)
        w = round((xmax - xmin) / width, 6)
        h = round((ymax - ymin) / height, 6)
        with open(os.path.join(newdir,label_name), 'a+') as f:
            f.write(' '.join([str(label), str(x_center), str(y_center), str(w), str(h) + '\n']))
    # except:
    #     shutil.move(txt_path, error_path)
print('ok')