import os
import xml.etree.ElementTree as ET

dirpath = 'E:/6/xml/'  # 原来存放xml文件的目录
newdir = 'E:/6/yolo/'  # 修改label后形成的txt目录

if not os.path.exists(newdir):
    os.makedirs(newdir)

for fp in os.listdir(dirpath):

    root = ET.parse(os.path.join(dirpath, fp)).getroot()  #?

    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    sz = root.find('size')
    width = float(sz[0].text)
    height = float(sz[1].text)
    filename = root.find('filename').text
    for child in root.findall('object'):  # 找到图片中的所有框
        # print(child.find('name').text)

        sub = child.find('bndbox')  # 找到框的标注值并进行读取
        label = child.find('name').text
        xmin = float(sub[0].text)

        ymin = float(sub[1].text)

        xmax = float(sub[2].text)

        ymax = float(sub[3].text)
        try:  # 转换成yolov3的标签格式，需要归一化到（0-1）的范围内
            x_center = round((xmin + xmax) / (2 * width),6)
            y_center = round((ymin + ymax) / (2 * height),6)
            w = round((xmax - xmin) / width,6)
            h = round((ymax - ymin) / height,6)
        except ZeroDivisionError:
            print(filename, '的 width有问题')

        with open(os.path.join(newdir, fp.split('.')[0] + '.txt'), 'a+') as f:
            f.write(' '.join([str(label), str(x_center), str(y_center), str(w), str(h) + '\n']))

print('ok')
