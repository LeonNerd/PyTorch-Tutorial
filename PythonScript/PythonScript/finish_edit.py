
from glob2 import glob
import os, os.path, shutil
import cv2
import numpy as np

# #创建文件夹判断
# def wcj(file,label):
#     isExists = os.path.exists('E:/1/'+label)
#     if not isExists:
#         os.makedirs('E:/1/'+label)
#     os.chdir('E:/1/'+label)                  #加入文件夹的路径
#     for JPG in file:
#         isExists = os.path.exists(JPG)
#         if not isExists:
#             os.makedirs(JPG)
# #创建文件夹
# os.chdir('E:/1/Images')  # 加入images的路径
# file = glob('*')
# wcj(file, 'zp1')
# wcj(file, 'yp1')
# wcj(file, 'fz1')
# os.chdir('E:/1/labels/')  # 加入label的路径
# files = glob('*')
# wcj(files, '1')
# wcj(files, '2')
# wcj(files, '3')

#图片变化
# os.walk('E:/1/')
# Root = 'E:/2/'
# Dest = 'E:/3/'
# for root, dirs, files in os.walk(Root):
#     new_root = root.replace(Root, Dest, 1)
#     if not os.path.exists(new_root):
#         os.mkdir(new_root)
#     for d in dirs:
#         d = os.path.join(new_root, d)
#         if not os.path.exists(d):
#             os.mkdir(d)
# #     for f in files:
#         # 把文件名分解为 文件名.扩展名
#         # 在这里可以添加一个 filter，过滤掉不想复制的文件类型，或者文件名
#         (shotname, extension) = os.path.splitext(f)
#         # 原文件的路径
#         old_path = os.path.join(root, f)
#         new_name = shotname + '_bak' + extension
#         # 新文件的路径
#         new_path = os.path.join(new_root, new_name)
#         try:
#             # 复制文件
#             open(new_path, 'wb').write(open(old_path, 'rb').read())
#         except IOError as e:
#             print(e)

# for root, dirs, files in os.walk('E:/1/'):
#     for name in files:
#         # jpg=os.path.join(root, name)
#         (shotname, extension) = os.path.splitext(name)       # 把文件名分解为 文件名.扩展名
#         if extension == '.jpg':
#             jpg = os.path.join(root, name)
#             print(jpg)
#             img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
#             # 读取文件大小
#             l = []
#             sp = img.shape
#             l.append(sp[1])  # width(colums) of image
#             imgInfo = sp
#             cols = imgInfo[0]
#             rows = imgInfo[1]
#             M = np.float32([[1, 0, 15], [0, 1, 0]])  # 左移15
#             dst = cv2.warpAffine(img, M, (rows, cols))
#             jpg1 =(os.path.abspath(os.path.join(os.getcwd(), jpg)))
#             cv2.imencode('.jpg', dst)[1].tofile(jpg1)  # 加入zp1的路径

'''
os.chdir('E:/1/Images/')  # 加入images的路径
files = glob('*/*.jpg')
for jpg in files:  # 确认文件格式
    img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
    #读取文件大小
    l = []
    sp = img.shape
    l.append(sp[1])  # width(colums) of image
    imgInfo = sp
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


# os.chdir('E:/1/labels/')  # 加入label的路径
# files = glob('*')
# wcj(files, '1')
# wcj(files, '2')
# wcj(files, '3')
# os.chdir('E:/1/Images/')  # 加入image的路径
# file = glob('*/*.jpg')
# for fn in file:
#     l = []
#     img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), -1)
#     sp = img.shape
#     l.append(sp[1])  # width(colums) of image

#txt write
os.chdir('E:/1/labels/')
files = glob('*/*.txt')
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
            f1.write(str(a) + ' ' + str(x1 - 15) + " " + str(y1) + ' ' + str(x2 - 15) + ' ' + str(y2) + '\n')
            f2.write(str(a) + ' ' + str(x1 + 15) + " " + str(y1) + ' ' + str(x2 + 15) + ' ' + str(y2) + '\n')
            f3.write(str(a) + ' ' + str(l[q]-x2) + " " + str(y1) + ' ' + str(l[q]-x1) + ' ' + str(y2) + '\n')
    f1.close()
    f2.close()
    f3.close()
    q += 1
'''
