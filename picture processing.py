import cv2
import os
from glob2 import glob
import os, os.path, shutil
import numpy as np
# def make_file():
    #批量复制创建文件夹，
os.getcwd()
os.chdir('E:/123456/Images')
file = glob('*')
print(file)
os.chdir('E:/123456/Images2')
for JPG in file:
    isExists = os.path.exists(JPG)
    if not isExists:
        os.makedirs(JPG)
#批量翻转图片存入已创建文件夹
os.getcwd()
os.chdir('E:/123456/Images')
files = glob('*/*.jpg')

root_path = "E:/123456/Images2/"
for jpg in files: #确认文件格式
    img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
    horizontal_img = cv2.flip(img, 1)
    splitName = jpg.split(".")
    newName = splitName[0]
    # cv2.imwrite(root_path+newName + '_flip.jpg', horizontal_img)
    cv2.imencode('.jpg', horizontal_img)[1].tofile(root_path+jpg)  # 保存图片



