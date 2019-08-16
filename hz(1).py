import sys
import codecs
from glob2 import glob
import os, os.path, shutil
import cv2
import numpy as np

#左平移15
#图左移
def pict_left_shift( ):

    # 批量图片存入已创建文件夹
    os.getcwd()
    os.chdir('E:/123456/Images')
    file = glob('*')
    #print(file)
    os.chdir('E:/123456/Images2')
    for JPG in file:
        isExists = os.path.exists(JPG)
        if not isExists:
            os.makedirs(JPG)
    # 批量平移图片存入已创建文件夹
    os.getcwd()
    os.chdir('E:/123456/Images')
    files = glob('*/*.jpg')

    root_path = "E:/123456/Images2/"

    for jpg in files:  # 确认文件格式
        img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
        imgInfo = img.shape
        cols = imgInfo[0]
        rows = imgInfo[1]

        # 平移矩阵M：[[1,0,x],[0,1,y]]
        M = np.float32([[1, 0, -15], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (rows, cols))
        # splitName = jpg.split(".")
        # newName = splitName[0]
        # cv2.imwrite(root_path+newName + '_flip.jpg', horizontal_img)
        cv2.imencode('.jpg', dst)[1].tofile(root_path + jpg)  # 保存图片

#标注文本左移
def txt_left_shift( ):
    #批量复制创建文件夹
    os.getcwd()
    os.chdir('E:/123456/labels/')
    files = glob('*')
   # print(files)
    os.chdir('E:/123456/labels2/')
    for txt in files:
        isExists = os.path.exists(txt)
        if not isExists:
            os.mkdir(txt)
    os.getcwd()
    os.chdir('E:/123456/labels/')
    files = glob('*/*.txt')
    for txt in files:
        f1 = open('E:/123456/labels2/' + txt, 'w')
        with open(txt)as f:  # 返回一个文件对象
            # lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
            line = f.readline()  # 以行的形式进行读取文件
            print(line)
            list1 = []
            list2 = []
            list3 = []
            while line:
                a = line.split()
                b = a[1:3]
                c = a[3:5]  # 这是选取需要读取的位数
                list3.append(a[0])
                list1.append(b)
                list2.append(c)  # 将其添加在列表之中
                line = f.readline()
            list1 = filter(None, list1)
            list1 = list(list1)
            list2 = filter(None, list2)
            list2 = list(list2)
            f1.write(str(len(list1)) + '\n')
            for i in range(len(list1)):
                list1[i][0] = int(list1[i][0]) + 15  # 左移-15，右移+15
                list1[i][1] = int(list1[i][1])
                x = tuple(list1[i])
                list2[i][0] = int(list2[i][0]) + 15
                list2[i][1] = int(list2[i][1])
                y = tuple(list2[i])
                print("x", x)
                print("y", y)
                print(str(list3[i]))
                f1.write(
                    str(list3[i + 1]) + ' ' + str(x[0]) + " " + str(x[1]) + ' ' + str(y[0]) + ' ' + str(y[1]) + '\n')
        f1.close()


#右平移15
#图右移
def pict_right_shift( ):
    os.getcwd()
    os.chdir('E:/123456/Images')
    file = glob('*')
    #print(file)
    os.chdir('E:/123456/Images3')
    for JPG in file:
        isExists = os.path.exists(JPG)
        if not isExists:
            os.makedirs(JPG)
    # 批量翻转图片存入已创建文件夹
    os.getcwd()
    os.chdir('E:/123456/Images')
    files = glob('*/*.jpg')

    root_path = "E:/123456/Images3/"
    for jpg in files:  # 确认文件格式
        img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
        # horizontal_img = cv2.flip(img, 1)
        imgInfo = img.shape
        cols = imgInfo[0]
        rows = imgInfo[1]
        M = np.float32([[1, 0, 15], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (rows, cols))
        # dst = str(dst)
        cv2.imencode('.jpg', dst)[1].tofile(root_path + jpg)   # 保存图片
#标注文本右移
def txt_right_shift( ):
    # 批量复制创建文件夹
    os.getcwd()
    os.chdir('E:/123456/labels/')
    files = glob('*')
    #print(files)
    os.chdir('E:/123456/labels3/')
    for txt in files:
        isExists = os.path.exists(txt)
        if not isExists:
            os.mkdir(txt)
    os.getcwd()
    os.chdir('E:/123456/labels/')
    files = glob('*/*.txt')
    for txt in files:
        f1 = open('E:/123456/labels3/' + txt, 'w')
        with open(txt)as f:  # 返回一个文件对象
            # lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
            line = f.readline()  # 以行的形式进行读取文件
            # print(lines)
            print(line)
            list1 = []
            list2 = []
            list3 = []
            while line:
                a = line.split()
                b = a[1:3]
                c = a[3:5]  # 这是选取需要读取的位数
                list3.append(a[0])
                list1.append(b)
                list2.append(c)  # 将其添加在列表之中
                line = f.readline()
            list1 = filter(None, list1)
            list1 = list(list1)
            list2 = filter(None, list2)
            list2 = list(list2)
            f1.write(str(len(list1)) + '\n')
            for i in range(len(list1)):
                list1[i][0] = int(list1[i][0]) + 15  # 左移-15，右移+15
                list1[i][1] = int(list1[i][1])
                x = tuple(list1[i])
                list2[i][0] = int(list2[i][0]) + 15
                list2[i][1] = int(list2[i][1])
                y = tuple(list2[i])
                print("x", x)
                print("y", y)
                print(str(list3[i]))
                f1.write(
                    str(list3[i + 1]) + ' ' + str(x[0]) + " " + str(x[1]) + ' ' + str(y[0]) + ' ' + str(y[1]) + '\n')
        f1.close()
#翻准
#图翻转
def pict_flip():
    # 批量复制创建文件夹
    os.getcwd()
    os.chdir('E:/123456/Images')
    file = glob('*')
   # print(file)
    os.chdir('E:/123456/Images4/')
    for JPG in file:
        isExists = os.path.exists(JPG)
        if not isExists:
            os.makedirs(JPG)

    # 批量翻转图片
    os.getcwd()
    os.chdir('E:/123456/Images')
    files = glob('*/*.jpg')

    root_path = "E:/123456/Images4/"
    for jpg in files:  # 确认文件格式
        img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
        horizontal_img = cv2.flip(img, 1)
        splitName = jpg.split(".")
        newName = splitName[0]
        # cv2.imwrite(root_path+newName + '_flip.jpg', horizontal_img)
        # cv2.imencode('.jpg',horizontal_img)[1].tofile(root_path)
        cv2.imencode('.jpg', horizontal_img)[1].tofile(root_path + jpg)

#标注文本翻转
def txt_flip():
    os.getcwd()
    os.chdir('E:/123456/labels/')
    files = glob('*')
    os.chdir('E:/123456/labels4/')
    for txt in files:
        isExists = os.path.exists(txt)
        if not isExists:
            os.mkdir(txt)
    os.getcwd()
    os.chdir('E:/123456/Images/')
    file = glob('*/*.jpg')

    for fn in file:  # 确认文件格式

        l = []
        img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), -1)
        sp = img.shape

        l.append(sp[1])  # width(colums) of image
        # print(l[fn])
    os.getcwd()
    os.chdir('E:/123456/labels/')
    files = glob('*/*.txt')

    for txt in files:
        q = 0
        f1 = open('E:/123456/labels4/' + txt, 'w')
        with open(txt)as f:  # 返回一个文件对象

            # lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
            line = f.readline()  # 以行的形式进行读取文件
            # print(lines)
            list1 = []
            list2 = []
            list3 = []
            while line:

                a = line.split()
                b = a[1:3]
                c = a[3:5]  # 这是选取需要读取的位数
                list3.append(a[0])
                list1.append(b)
                list2.append(c)  # 将其添加在列表之中
                line = f.readline()

            list1 = filter(None, list1)
            list1 = list(list1)
            list2 = filter(None, list2)
            list2 = list(list2)
            f1.write(str(len(list1)) + '\n')

            for i in range(len(list1)):

                x2 = list1[i][0] = l[q] - int(list1[i][0])
                y1 = list1[i][1] = int(list1[i][1])
                # x = tuple(list1[i])
                x1 = list2[i][0] = l[q] - int(list2[i][0])
                y2 = list2[i][1] = int(list2[i][1])
                # y = tuple(list2[i])
                # print("x", x)
                # print("y", y)
                # print(str(list3[i]))
                f1.write(str(list3[i + 1]) + ' ' + str(x1) + " " + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')
        f1.close()
        q += 1

if __name__ == '__main__':
    pict_left_shift()
    txt_left_shift()
    pict_right_shift()
    txt_right_shift()
    pict_flip()
    txt_flip()
