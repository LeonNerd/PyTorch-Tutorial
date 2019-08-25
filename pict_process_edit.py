import sys
import codecs
from glob2 import glob
import os, os.path, shutil
import cv2
import numpy as np
# 主函数-创建文件夹
def judge_creat(file,label):
    isExists = os.path.exists('E:/1/'+label)
    if not isExists:
        os.makedirs('E:/1/'+label)
    os.chdir('E:/1/'+label)                  #加入文件夹的路径
    for JPG in file:
        isExists = os.path.exists(JPG)
        if not isExists:
            os.makedirs(JPG)
def creat_folder():
    # 存储图片文件夹
    os.chdir('E:/1/Images')  # 读取原文件 Images 为根文件夹
    file = glob('*')
    judge_creat(file, 'pict_left_shift')
    judge_creat(file, 'pict_right_shift')
    judge_creat(file, 'pict_flip')
    os.chdir('E:/1/labels/')  # 加入label的路径
    files = glob('*')
    judge_creat(files, 'txt_left_shift')
    judge_creat(files, 'txt_right_shift')
    judge_creat(files, 'txt_flip')
# 主函数-图片移动
def pict_transform():
    os.chdir('E:/1/Images')  # 读取原文件夹下的图片
    files = glob('*/*.jpg')  # 判断是否是jpg格式图片
    pict_left_shift(files)  # 图片左移，如不需要注释
    pict_right_shift(files)  # 图片右移动，如不需要注释
    pict_flip(files)  # 图片翻转
# 主函数-移动后的图片坐标信息写入
def txt_write():
    os.chdir('E:/1/Images/')  # 加入Images路径
    file = glob('*/*.jpg')
    l = []
    for fn in file:  # 确认文件格式
        
        img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), -1)
        shape = img.shape
        l.append(shape[1])
    os.chdir('E:/1/labels/')
    files1 = glob('*/*.txt')
    txt_left_shift(files1)
    txt_right_shift(files1)
    txt_flip(files1,l)

def pict_left_shift(files):
    root_path = "../pict_left_shift/"
    for jpg in files:  # 确认文件格式
        img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
        imgInfo = img.shape
        cols = imgInfo[0]
        rows = imgInfo[1]
        # 平移矩阵M：[[1,0,x],[0,1,y]]
        M = np.float32([[1, 0, -15], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (rows, cols))
        cv2.imencode('.jpg', dst)[1].tofile(root_path + jpg)  # 保存图片
def pict_right_shift(files):
    root_path = "../pict_right_shift/"
    for jpg in files:  # 确认文件格式
        img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
        imgInfo = img.shape
        cols = imgInfo[0]
        rows = imgInfo[1]
        M = np.float32([[1, 0, 15], [0, 1, 0]])  # 1 0: x   0 1: y;  如需修改移动像素只需在第三位修改 （-15为左移15）
        dst = cv2.warpAffine(img, M, (rows, cols))
        cv2.imencode('.jpg', dst)[1].tofile(root_path + jpg)  # 保存图片
def pict_flip(files):
    root_path = "../pict_flip/"
    for jpg in files:  # 确认文件格式
        img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)  # 读取图片（中英文命名皆可）
        horizontal_img = cv2.flip(img, 1)  # 1 水平翻转； 0 垂直翻转；  -1 水平垂直翻转（旋转180）
        cv2.imencode('.jpg', horizontal_img)[1].tofile(root_path + jpg)  # 写入图片（中英文路径皆可）root_path为写入路径

def txt_left_shift(files1):  # 左移后图片坐标信息写入
    for txt in files1:
        f1= open('../txt_left_shift/' + txt, 'w')       #开启'w'写模式
        with open(txt)as f:  # 返回一个文件对象
            list = f.readline()
            f1.write(list[0] + '\n')
            for i in range(int(list[0])):
                list1 = f.readline()
                a = int(list1.split(' ')[0])
                x1 = int(list1.split(' ')[1])
                y1 = int(list1.split(' ')[2])
                x2 = int(list1.split(' ')[3])
                y2 = int(list1.split(' ')[4])
                f1.write(str(a) + ' ' + str(x1 - 15) + " " + str(y1) + ' ' + str(x2 - 15) + ' ' + str(y2) + '\n')
        f1.close()
def txt_right_shift(files1):
    for txt in files1:
        f2 = open('../txt_right_shift/' + txt, 'w')       #开启'w'写模式
        with open(txt)as f:  # 返回一个文件对象
            list = f.readline()
            f2.write(list)
            for i in range(int(list)):
                list1 = f.readline()
                a = int(list1.split(' ')[0])
                x1 = int(list1.split(' ')[1])
                y1 = int(list1.split(' ')[2])
                x2 = int(list1.split(' ')[3])
                y2 = int(list1.split(' ')[4])
                f2.write(str(a) + ' ' + str(x1 + 15) + " " + str(y1) + ' ' + str(x2 + 15) + ' ' + str(y2) + '\n')
        f2.close()
def txt_flip(files1,l):
    for txt in files1:
        q = 0
        f3 = open('../txt_flip/' + txt, 'w')
        with open(txt)as f:  # 返回一个文件对象
            list = f.readline()
            f3.write(list[0] + '\n')
            for i in range(int(list[0])):
                list1 = f.readline()
                a = int(list1.split(' ')[0])
                x1 = int(list1.split(' ')[1])
                y1 = int(list1.split(' ')[2])
                x2 = int(list1.split(' ')[3])
                y2 = int(list1.split(' ')[4])
                f3.write(str(a) + ' ' + str(l[q] - x2) + " " + str(y1) + ' ' + str(l[q] - x1) + ' ' + str(y2) + '\n')
            f3.close()
            q += 1
if __name__ == '__main__':
    creat_folder()
    print('creat_folder finish')
    pict_transform()
    print('pict_transform finish')
    txt_write()
    print('txt_write finish')

