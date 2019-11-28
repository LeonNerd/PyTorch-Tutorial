import cv2
import os.path

# 使用opencv按一定间隔截取视频帧，并保存为图片

filepath = r'G:/1116_3/'  # 视频所在文件夹
pathDir = os.listdir(filepath)
# a = 1  # 图片计数
for allDir in pathDir:
    # a = 1
    videopath = r'G:/1116_3/' + allDir
    print(videopath)

    vc = cv2.VideoCapture(videopath) # 读入视频文件
    # vc = cv2.VideoCapture.open(videopath)

    c = 1

    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()

    else:
        rval = False
    a = 1
    timeF = 130   # 视频帧计数间隔频率

    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (c % timeF == 0):  # 每隔timeF帧进行存储操作
            # video_to_picture_path = os.path.join(save_path, item.split(".")[0])  # 视频文件夹的命名
            # if not os.path.exists(video_to_picture_path):  # 创建每一个视频存储图片对应的文件夹
            #     os.makedirs(video_to_picture_path)
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame,0)
            cv2.imencode('.jpg', gray)[1].tofile(r'G:/1116_4/' + allDir+'_' + str(a) + '.jpg')  # 路径含中文存图
            # cv2.imwrite('F:/项目数据集/微光相机视频/0918/' +
            # str(a) + '.jpg', gray)


            a = a + 1

        c = c + 1
        cv2.waitKey(1)

    vc.release()
    cv2.destroyAllWindows()