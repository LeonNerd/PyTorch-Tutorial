#此代码可以将xml文件格式转换为yolo格式
import os
import glob
import shutil
import xml.etree.ElementTree as ET
label_dict = {'person':0,'car':1,'TaDiao':2,'DiaoChe':3,'ShiGongJiXie':4,'aqwl':5}
dirpath =  'H:/data/VOCdevkit/voc/2/'  # 原来存放xml文件的目录
newdir = 'H:/data/VOCdevkit/voc/3'  # 修改label后形成的txt目录
errorpath = "H:/data/VOCdevkit/voc/4"
if not os.path.exists(newdir):
    os.makedirs(newdir)
os.chdir(dirpath)
for fp in glob.glob('*.xml'):
    label_name = fp.split('.')[0]+ '.txt'
    print(label_name)

    txt_path = dirpath + fp.split('.')[0] + '.txt'
    error_txt_path = errorpath + fp.split('.')[0] + '.txt'


    root = ET.parse(os.path.join(dirpath, fp)).getroot()  #?
    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    sz = root.find('size')
    width = float(sz[0].text)
    height = float(sz[1].text)
    filename = root.find('filename').text
    # print(fp)

    for child in root.findall('object'):  # 找到图片中的所有框
        sub = child.find('bndbox')  # 找到框的标注值并进行读
        # print(child.find('name').text)
        label = label_dict[child.find('name').text]
        xmin = float(sub[0].text)
        ymin = float(sub[1].text)
        xmax = float(sub[2].text)
        ymax = float(sub[3].text)
        try:  # 转换成yolov3的标签格式，需要归一化到（0-1）的范围内
            x_center = round((xmin + xmax) / (2 * width), 6)
            y_center = round((ymin + ymax) / (2 * height), 6)
            w = round((xmax - xmin) / width, 6)
            h = round((ymax - ymin) / height, 6)
        # except ZeroDivisionError:
        #     print(filename, '的 width有问题')
        except ET.ParseError:
            shutil.copy(  txt_path , error_txt_path)

        with open(os.path.join(newdir,label_name), 'a+') as f:
            f.write(' '.join([str(label), str(x_center), str(y_center), str(w), str(h) + '\n']))

print('ok')