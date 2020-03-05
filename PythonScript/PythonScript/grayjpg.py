# -*- coding:utf-8 -*-
import cv2
import os

input_dir = r'F:\0214\pict/'
out_dir = r'F:\0214\pict_jpg/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

a = os.listdir(input_dir)
for i in a:
    print(i)
    # img = cv2.imread(input_dir+i)
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.imencode('.jpg', gray)[1].tofile(out_dir+i)
    try:
        img = cv2.imread(input_dir + i)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imencode('.jpg', gray)[1].tofile(out_dir + i)
    except:
        pass
#     continue


#
# import cv2
# #循环灰度图片并保存
# def grayImg():
#     for x in range(1,38):
#         #读取图片
#         img = cv2.imread("C:/Users/11473/Desktop/army_tx/jpg/{}.jpg".format(str(x)))
#         GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         #保存灰度后的新图片
#         cv2.imwrite("C:/Users/11473/Desktop/army_tx/gray/{}.jpg".format(str(x)),GrayImage)
# grayImg()


# from PIL import Image
# import os
# #
# input_dir = 'C:/Users/11473/Desktop/army_tx/jpg/'
# out_dir = 'C:/Users/11473/Desktop/army_tx/gray/'
# a = os.listdir(input_dir)
# for i in a:
#     I = Image.open('input_dir + i+.jpg')
#     L = I.convert('L')
#     L.save('out_dir + i+.jpg')


# from PIL import Image
# import os
#
#
# #灰度化
#
#
# infile = 'C:/Users/11473/Desktop/jpg/' #原始图像路径
# outfile= 'C:/Users/11473/Desktop/gray2/' #灰度化后的图像路径
# a = os.listdir(infile)
# for i in a:
#     print(i)
#     im = Image.open(infile + i).convert('L')  # 灰度化
#     im.save(outfile + i)  # 存储图片
#     # try:
#     #     ImportError
#     # except IndexError as e:
#     #     print("IndexError Details : " + str(e))
# pass
    # continue
