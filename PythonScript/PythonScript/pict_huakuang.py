import codecs
import cv2
import os
from glob2 import glob

f = codecs.open('H:/22.txt', mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
img = cv2.imread("H:/22.jpg")
line = f.readline()   # 以行的形式进行读取文件
list1 = []
list2 = []
while line:
    a = line.split()
    b = a[1:3]  # 这是选取需要读取的位数
    c = a[3:5]
    list1.append(b)
    list2.append(c)   # 将其添加在列表之中
    line = f.readline()
f.close()
list1 = filter(None, list1)
list1 = list(list1)
list2 = filter(None, list2)
list2 =list(list2)
for i in range(len(list1)):
    list1[i][0] = int(list1[i][0])
    list1[i][1]=int(list1[i][1])
    x=tuple(list1[i])
    list2[i][0] = int(list2[i][0])
    list2[i][1] = int(list2[i][1])
    y = tuple(list2[i])
    # print(x,y)


# 画矩形框
    cv2.rectangle(img, ( x), (y ), (0, 255, 0), 4)
    cv2.imshow("image", img)
cv2.waitKey(0)



