# coding:utf-8
from __future__ import print_function
from glob2 import glob
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
import os
from PIL import Image,ImageEnhance


def decode(im):
    # 在这里查找二维码
    decodedObjects = pyzbar.decode(im)
    # print(type(decodedObjects))
    # 打印出结果
    for obj in decodedObjects:

        # print('Name : ', obj)

        print('Type : ', obj.type)
        print('Data : ', obj.data.decode("utf-8"), '\n')






    return decodedObjects



def display(im, decodedObjects):

    for decodedObject in decodedObjects:
        points = decodedObject.polygon

        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points;


        n = len(hull)

        for j in range(0, n):
            cv2.line(im, hull[j], hull[(j + 1) % n], (255, 0, 0), 3)

    cv2.imshow("Results", im);
    cv2.waitKey(0);

# Read image
os.chdir('E:/1/')  # 加入Images路径
files = glob('*')
i=0
for file in files:
    i = i + 1
    print(i)
    print(file)
    im = cv2.imread(file)
    decodedObjects = decode(im)
    result = decodedObjects
    if len(result) == 0:
        img = Image.open(file)
        # img = ImageEnhance.Brightness(img).enhance(2.0)#增加亮度
        # img = ImageEnhance.Sharpness(img).enhance(17.0)#锐利化
        # img = ImageEnhance.Contrast(img).enhance(4.0)#增加对比度
        # img = img.convert('L')#灰度化
        # im.show()
        barcodes = pyzbar.decode(img)
        if len(barcodes) == 0:
            imgInfo = im.shape
            height = imgInfo[0]
            width = imgInfo[1]
            mode = imgInfo[2]
            # 1 放大 缩小 2 等比例 非等比例
            dstHeight = int(height * 1.5)
            dstWeight = int(width * 1.5)
            # 最近邻域插值 双线性插值 像素关系重采样 立方插值
            dst = cv2.resize(im, (dstWeight, dstHeight))
            decodedObjects = decode(dst)



        # for barcode in barcodes:
        #     barcodeData = barcode.data.decode("utf-8")
        #     print(len(barcodeData))

    display(im, decodedObjects)
