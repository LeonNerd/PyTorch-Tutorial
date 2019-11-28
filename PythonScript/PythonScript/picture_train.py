# import cv2 as cv
# import cv2
# import os, os.path, shutil
# from glob2 import glob
# import numpy as np
# import matplotlib.pyplot as plt
#
# img = cv.imread("E:/1/Images/278/2019082119175.jpg",)
#
# # 计算灰度直方图  Matplotlib本身也提供了计算直方图的函数hist，以下由matplotlib实现直方图的生成：
#
# h, w = img.shape[:2]
#
# pixelSequence = img.reshape([h * w, ])
#
# numberBins = 256
# histogram, bins, patch = plt.hist(pixelSequence, numberBins,facecolor='black', histtype='bar')
# plt.xlabel("gray label")
# plt.ylabel("number of pixels")
# plt.axis([0, 255, 0, np.max(histogram)])
# plt.show()
# cv.imshow("img", img)
# cv.waitKey()
#
# # #线性变换
# out = 2.0 * img
# # 进行数据截断，大于255的值截断为255
# out[out > 255] = 250
# # 数据类型转换
# out = np.around(out)
# out = out.astype(np.uint8)
# cv.imshow("img", img)
# cv.imshow("out", out)

#直方图正规化:
# 计算原图中出现的最小灰度级和最大灰度级
# 使用函数计算
# Imin, Imax = cv.minMaxLoc(img)[:2]
# Omin, Omax = 0, 255
# # 计算a和b的值
# a = float(Omax - Omin) / (Imax - Imin)
# b = Omin - a * Imin
# out = a * img + b
# out = out.astype(np.uint8)
# cv.imshow("img", img)
# cv.imshow("out", out)

#正规化函数normalize:
# out = np.zeros(img.shape, np.uint8)
# cv.normalize(img, out, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
# cv.imshow("img", img)
# cv.imshow("out", out)



# width,height = img.shape[:2]
# print(width)
#
# res2 = cv.resize(img, (int(1.8*width), int(0.4*height)), interpolation=cv.INTER_AREA)
# cv.imshow('origin_picture', img)
# cv.imshow('res2', res2)
# BLUE = [0,0,0]
#
# replicate = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REPLICATE)   #重复，就是对边界的像素进行复制
# reflect = cv2.copyMakeBorder(img,300,100,100,100,cv2.BORDER_REFLECT)
# reflect101 = cv2.copyMakeBorder(img,100,100,100,100,cv2.BORDER_REFLECT_101)
# wrap = cv2.copyMakeBorder(img,80,80,80,100,cv2.BORDER_WRAP)
# constant= cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)  #有几个像素宽的边界
# cv.imshow('origin_picture', img)
# cv.imshow('res2',replicate)
#
# cv.waitKey()



# import cv2
# import numpy as np
# img = cv.imread("E:/untitled/20190812/Images/O/B_3436.jpg",)
# cols,rows= img.shape[:2]
# # 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
# # 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
# M = cv2.getRotationMatrix2D((rows / 2, cols / 2), 90, 0.5)
# dst = cv2.warpAffine(img, M, (rows, cols))
# cv2.imshow('img1',dst)
# cv2.waitKey(0)
# #

# import cv2
# import numpy as np
# os.chdir('E:/1/Images')  # 读取原文件夹下的图片
# files = glob('*/*.jpg')  # 判断是否是jpg格式图片
# root_path = "../resize_pict/"
# for jpg in files:  # 确认文件格式
#     img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
#     # 下面的 None 本应该是输出图像的尺寸，但是因为后边我们设置了缩放因子
#     # 因此这里为 None
#     res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#     #or
#     # 这里呢，我们直接设置输出图像的尺寸，所以不用设置缩放因子
#     height, width = img.shape[:2]       #提取图片宽高
# # #推荐的插值方法是缩小时用cv2.INTER_AREA，放大用cv2.INTER_CUBIC(慢)和cv2.INTER_LINEAR。
# #默认情况下差值使用cv2.INTER_LINEAR。
#     resize = cv2.resize(img, (int(0.4*width), int(0.4*height)), interpolation=cv2.INTER_AREA)  #修改图片大小
#     cv2.imencode('.jpg', resize)[1].tofile(root_path + jpg)  # 保存图片



# import cv2
# import numpy as np
# img = cv2.imread("E:/1/pict_flip/278/mgF0 (5688).jpg") #读取图片地址
# #注意img.shape[0]是图片高度
# cols,rows  = img.shape[:2]
# M = np.float32([[1, 0, 100], [0, 1, 50]])   # 平移矩阵M：[[1,0,x],[0,1,y]]
# #调用warpAffine函数，仿射出修改后图片的矩阵
# #CV2.warpAffine(原图，修改后图，图片大小（先宽，后高）)
# dst = cv2.warpAffine(img, M, (rows, cols))
# cv2.imshow('img1',dst)
# cv2.waitKey(0)
#
# import cv2
# import numpy as np
# img = cv2.imread("mgF0 (5688).jpg") #读取图片地址
# h_flip = cv2.flip(img, 1)   #水平翻转为1，垂直翻转为0， -1表示水平垂直翻转
# cv2.imwrite('mgF0 (5688).jpg',h_flip )



