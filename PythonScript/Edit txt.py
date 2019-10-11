#更改Labels中类别号
from glob2 import glob
import os, os.path, shutil
import cv2
import numpy as np

# root_path='C:/Users/11473/Desktop/test/'
# def wcj(file,label):
#     isExists = os.path.exists(root_path+label)
#     # if not isExists:
#     #     os.makedirs(root_path+label)
#     os.chdir(root_path+label)                  #加入文件夹的路径
#     for txt in file:
#         isExists = os.path.exists(txt)
#         if not isExists:
#             os.makedirs(txt)
# os.chdir(root_path)  # 加入label的路径
# files = glob('*')
# wcj(files, '1')

os.chdir('H:/data/Yan1/Labels1/')
files = glob('*.txt')
for txt in files:
    f1 = open('H:/data/Yan1/Labels2/'+ txt, 'w')
    # jpg = jpg.replace('\\', '/')
    # q = 0
    with open('H:/data/Yan1/Labels1/' + txt)as f:  # 返回一个文件对象
            list=f.readlines()
            for line in list:
                a = line.split(' ')[0]
                print(a)
                if a == '0':
                    a1=2
                    x1 = line.split(' ')[1]
                    y1 = line.split(' ')[2]
                    x2 = line.split(' ')[3]
                    y2 = line.split(' ')[4]
                    f1.write(str(a1) + ' ' + str(x1) + " " + str(y1) + ' ' + str(x2) + ' ' + str(y2) )
                elif a == '2':
                    a2 = 1
                    x1 = line.split(' ')[1]
                    y1 = line.split(' ')[2]
                    x2 = line.split(' ')[3]
                    y2 = line.split(' ')[4]
                    f1.write(str(a2) + ' ' + str(x1) + " " + str(y1) + ' ' + str(x2) + ' ' + str(y2) )
                else:
                    f1.write(line)
    #
    f1.close()


# print('标注文本平移翻转已完成，保存在1中')