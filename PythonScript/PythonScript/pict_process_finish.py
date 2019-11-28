from glob2 import glob
import os, os.path, shutil
import cv2
import numpy as np
#创建文件夹判断
def wcj(file,label):
    isExists = os.path.exists('H:/'+label)
    if not isExists:
        os.makedirs('H:/'+label)
    # os.chdir('H:/data/cover/'+label)                  #加入文件夹的路径
    # for JPG in file:
    #     isExists = os.path.exists(JPG)
    #     if not isExists:
    #         os.makedirs(JPG)
#图片变换
os.chdir('H:/')  # 加入images的路径
file = glob('*')
wcj(file, 'zp1')
wcj(file, 'yp1')
wcj(file, 'fz1')
os.chdir('H:/test3/')  # 加入images的路径
files = glob('*.jpg')
for jpg in files:  # 确认文件格式
    # img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
    img = cv2.imread(jpg)
    imgInfo = img.shape
    cols = imgInfo[0]
    rows = imgInfo[1]
    M = np.float32([[1, 0, -15], [0, 1, 0]])  # 左移15
    dst = cv2.warpAffine(img, M, (rows, cols))
    jpg = jpg.replace('\\', '/')
    cv2.imencode('.jpg', dst)[1].tofile('../zp1/' + jpg)  # 加入zp1的路径
    M2 = np.float32([[1, 0, 15], [0, 1, 0]])  # 右移15
    dst2 = cv2.warpAffine(img, M2, (rows, cols))
    cv2.imencode('.jpg', dst2)[1].tofile('../yp1/' + jpg)  # 加入yp1的路径
    horizontal_img = cv2.flip(img, 1)  # 翻转
    cv2.imencode('.jpg', horizontal_img)[1].tofile('../fz1/' + jpg)  # 加入fz1的路径
os.chdir('H:/')  # 加入label的路径
files = glob('*')
wcj(files, '1')
wcj(files, '2')
wcj(files, '3')
os.chdir('H:/test3/')  # 加入image的路径
file = glob('*.jpg')
for fn in file:
    l = []
    img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), -1)
    sp = img.shape
    l.append(sp[1])  # width(colums) of image
os.chdir('H:/test3/')
files = glob('*.txt')
for txt in files:
    f1 = open('../1/' + txt, 'w')
    f2 = open('../2/' + txt, 'w')
    f3 = open('../3/' + txt, 'w')
    q = 0
    with open(txt)as f:  # 返回一个文件对象
        list =f.readline()
        f1.write(list[0]+ '\n')
        f2.write(list[0] + '\n')
        f3.write(list[0] + '\n')
        for i in range(int(list[0])):
            list1 = f.readline()
            a = int(list1.split(' ')[0])
            x1 = int(list1.split(' ')[1])
            y1 = int(list1.split(' ')[2])
            x2 = int(list1.split(' ')[3])
            y2 = int(list1.split(' ')[4])
            print(x1,y1,x2,y2)
            f1.write(str(a) + ' ' + str(x1 - 15) + " " + str(y1) + ' ' + str(x2 - 15) + ' ' + str(y2) + '\n')
            f2.write(str(a) + ' ' + str(x1 + 15) + " " + str(y1) + ' ' + str(x2 + 15) + ' ' + str(y2) + '\n')
            f3.write(str(a) + ' ' + str(l[q]-x2) + " " + str(y1) + ' ' + str(l[q]-x1) + ' ' + str(y2) + '\n')
    f1.close()
    f2.close()
    f3.close()
    q += 1
