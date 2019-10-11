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
    # judge_creat(file, 'resize_pict')
    judge_creat(file, 'pict_flip')
    os.chdir('E:/1/labels/')  # 加入label的路径
    files = glob('*')
    judge_creat(files, 'txt_left_shift')
    judge_creat(files, 'txt_right_shift')
    judge_creat(files, 'txt_flip')


# 主函数-图片移动
def pict_transform():
    os.chdir('E:/1/Images')  # 读取原文件夹下的图片
    files = glob('*/*.png')  # 判断是否是jpg格式图片
    pict_left_shift(files)  # 图片左移，如不需要注释
    pict_right_shift(files)  # 图片右移动，如不需要注释
    pict_flip(files)  # 图片翻转
    # resize_pict(files)     #图片批量缩放
    # rotate_pict(files)      #图片批量旋转
    # copyMakeBorder(files)   #图片批量加边框


# 主函数-移动后的图片坐标信息写入
def txt_write():
    os.chdir('E:/1/Images/')  # 加入Images路径
    file = glob('*/*.png')
    l = []
    for fn in file:  # 确认文件格式
        img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), -1)
        shape = img.shape
        l.append(shape[1])
    os.chdir('E:/1/labels')
    files1 = glob('*/*.txt')
    # txt_left_shift(files1)
    # txt_right_shift(files1)
    # txt_flip(files1,l)
    txt_edit(files1, l)


# 子函数
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
        # splitName = jpg.split("/")
        # newName = splitName[1]
        # newName1 = newName.split('.')
        # newName2 = newName1[0]
        #
        # cv2.imwrite(root_path + '/' + newName2 + '_'+ '.jpg', dst)
        cv2.imencode('.png', dst)[1].tofile(root_path + jpg)  # 保存图片

def pict_right_shift(files):
    root_path = "../pict_right_shift/"
    for jpg in files:  # 确认文件格式
        img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
        imgInfo = img.shape
        cols = imgInfo[0]
        rows = imgInfo[1]
        M = np.float32([[1, 0, 15], [0, 1, 0]])  # 1 0: x   0 1: y;  如需修改移动像素只需在第三位修改 （-15为左移15）
        dst = cv2.warpAffine(img, M, (rows, cols))
        cv2.imencode('.png', dst)[1].tofile(root_path + jpg)  # 保存图片
def pict_flip(files):
    root_path = "../pict_flip/"
    for jpg in files:  # 确认文件格式
        img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)  # 读取图片（中英文命名皆可）
        horizontal_img = cv2.flip(img, 1)  # 1 水平翻转； 0 垂直翻转；  -1 水平垂直翻转（旋转180）
        cv2.imencode('.png', horizontal_img)[1].tofile(root_path + jpg)  # 写入图片（中英文路径皆可）root_path为写入路径
def resize_pict(files):
    root_path = "../resize_pict/"
    for jpg in files:  # 确认文件格式
        img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
        height, width = img.shape[:2]       #提取图片宽高
# #推荐的插值方法是缩小时用cv2.INTER_AREA，放大用cv2.INTER_CUBIC(慢)和cv2.INTER_LINEAR。
#默认情况下差值使用cv2.INTER_LINEAR。
        resize = cv2.resize(img, (int(0.4*width), int(0.4*height)), interpolation=cv2.INTER_AREA)  #修改图片大小
        cv2.imencode('.jpg', resize)[1].tofile(root_path + jpg)  # 保存图片
def rotate_pict(files):
    root_path = "../rotate_pict/"
    for jpg in files:  # 确认文件格式
        img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
        cols, rows = img.shape[:2]
        # 第一个参数是旋转中心，第二个参数是旋转角度，第三个参数是缩放比例
        M = cv2.getRotationMatrix2D((rows / 2, cols / 2), 45, 0.5)
        dst = cv2.warpAffine(img, M, (rows, cols))
        cv2.imencode('.png', dst)[1].tofile(root_path + jpg)  # 保存图片
def copyMakeBorder(files):
    root_path = "../resize_pict/"
    BLUE = [0,0,0]
    for jpg in files:  # 确认文件格式
        img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
        constant = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=BLUE)  # 有几个像素宽的边界
        cv2.imencode('.jpg', constant)[1].tofile(root_path + jpg)  # 保存图片

def txt_left_shift(files1):  # 左移后图片坐标信息写入
    for txt in files1:

        f1= open('../txt_left_shift/' + txt, 'w')       #开启'w'写模式
        with open(txt)as f:  # 返回一个文件对象
            list = f.readline()
            f1.write(list)
            for i in range(int(list)):
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
            f3.write(list)
            for i in range(int(list)):
                list1 = f.readline()
                a = int(list1.split(' ')[0])
                x2 = int(list1.split(' ')[1])
                y1 = int(list1.split(' ')[2])
                x1 = int(list1.split(' ')[3])
                y2 = int(list1.split(' ')[4])
                f3.write(str(a) + ' ' + str(l[q] - x1) + " " + str(y1) + ' ' + str(l[q] - x2) + ' ' + str(y2) + '\n')
        f3.close()
        q += 1
def txt_edit(files1,l):
    for txt in files1:
        q = 0
        f3 = open('../txt_flip/' + txt, 'w')
        with open(txt)as f:  # 返回一个文件对象
            list = f.readline()
            f3.write(list)
            for i in range(int(list)):
                list1 = f.readline()
                a = int(list1.split(' ')[0])
                x1 = int(list1.split(' ')[1])
                y1 = int(list1.split(' ')[2])
                x2 = int(list1.split(' ')[3])
                y2 = int(list1.split(' ')[4])
                if a == 1:
                    a = 2
                    f3.write(str(a) + ' ' + str(x1) + " " + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')
                elif a == 2:
                    a = 1
                    f3.write(str(a) + ' ' + str(x1) + " " + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')
        f3.close()
        q += 1
def edit_name():
    path = 'E:/1/2/'
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.split('.')[1] == 'txt':
                filename1 = filename.split('.')[0]
                newName = 'batch3_' + filename1 + '_flip'+ '.txt'
                os.rename(root +'/'+filename, root +'/'+newName)
                print("重命名 【%s】为【%s】成功! " % (filename, newName))
            else:
                print(None)
    # 获取指定路径的所有文件名字
    # filenames = os.listdir(path)
    # for filename in filenames:
    #     filename1 = filename.split('.')[0]
    #     # 因为获取到的文件名是str类型，利用拼接特性添加前缀
    #     newName = 'batch3_' + filename1 + '_flip'+ '.txt'
    #     os.rename(path + filename,  path + newName)
    #     print("重命名 【%s】为【%s】成功! " % (filename, newName))
def video_process():
    import cv2
    import os
    save_path = r"H:/03_1/"  # 存储的位置
    path = r"H:/03/"  # 要截取视频的文件夹
    filelist = os.listdir(path)  # 读取文件夹下的所有文件
    for item in filelist:
        if item.endswith('.mp4'):  # 根据自己的视频文件后缀来写，我的视频文件是mp4格式
            try:
                src = os.path.join(path, item)
                vid_cap = cv2.VideoCapture(src)  # 传入视频的路径
                if vid_cap.isOpened():  # 判断是否正常打开
                    success, image = vid_cap.read()
                else:
                    success = False
                count = 0
                c = 1
                timeF = 130  # 视频帧计数间隔频率
                while success:
                    success, image = vid_cap.read()
                    if (c % timeF == 0):  # 每隔timeF帧进行存储操作
                        # video_to_picture_path = os.path.join(save_path, item.split(".")[0])  # 视频文件夹的命名
                        # if not os.path.exists(video_to_picture_path):  # 创建每一个视频存储图片对应的文件夹
                        #     os.makedirs(video_to_picture_path)
                        cv2.imwrite(save_path + item.split(".")[0] + '_' + str(count) + ".jpg",
                                    image)  # 存储图片的地址 以及对图片的命名
                        count += 1
                    c += 1
                print('Total frames: ', count)  # 打印截取的图片数目
            except:
                print("error")

def huakuang():
    f = codecs.open('H:/labels_transform/batch3_0_1_flip.txt', mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
    img = cv2.imread("H:/photo_transform/batch10_0_1_flip.png")
    line = f.readline()  # 以行的形式进行读取文件
    list1 = []
    list2 = []
    while line:
        a = line.split()
        b = a[1:3]  # 这是选取需要读取的位数
        c = a[3:5]
        list1.append(b)
        list2.append(c)  # 将其添加在列表之中
        line = f.readline()
    f.close()
    list1 = filter(None, list1)
    list1 = list(list1)
    list2 = filter(None, list2)
    list2 = list(list2)
    for i in range(len(list1)):
        list1[i][0] = int(list1[i][0])
        list1[i][1] = int(list1[i][1])
        x = tuple(list1[i])
        list2[i][0] = int(list2[i][0])
        list2[i][1] = int(list2[i][1])
        y = tuple(list2[i])
        # print(x, y)

        # 画矩形框
        cv2.rectangle(img, (x), (y), (0, 255, 0), 4)
        cv2.imshow("image", img)
    cv2.waitKey(0)

if __name__ == '__main__':
    creat_folder()
    print('creat_folder finish')

    # pict_transform()
    # print('pict_transform finish')

    txt_write()
    print('txt_write finish')

    # edit_name()
    # print('edit_name finish')

    # video_process()
    # print('video_process finish')
    #
    # huakuang()
    # print('huakuang finish')