import cv2
import os
save_path=r"H:/03_1"      #存储的位置
path = r"H:/03/"   #要截取视频的文件夹
filelist = os.listdir(path) #读取文件夹下的所有文件
for item in filelist:
    if item.endswith('.mp4'):     #根据自己的视频文件后缀来写，我的视频文件是mp4格式
        try:
            src = os.path.join(path, item)
            vid_cap = cv2.VideoCapture(src)    #传入视频的路径
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
                    video_to_picture_path = os.path.join(save_path, item.split(".")[0])  # 视频文件夹的命名
                    if not os.path.exists(video_to_picture_path):  # 创建每一个视频存储图片对应的文件夹
                        os.makedirs(video_to_picture_path)
                    cv2.imwrite(video_to_picture_path + "/" + str(count) + ".jpg", image)  # 存储图片的地址 以及对图片的命名
                    count += 1
                c += 1
            print('Total frames: ', count)     #打印截取的图片数目
        except:
            print("error")