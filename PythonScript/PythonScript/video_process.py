import cv2
import os.path

# 使用opencv按一定间隔截取视频帧，并保存为图片

filepath = r'G:/1116_3/'  # 视频所在文件夹
pathDir = os.listdir(filepath)
i = 0
a = 1  # 图片计数
def count(videopath):


    video = cv2.VideoCapture(videopath);

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)

    else:
        fps = video.get(cv2.CAP_PROP_FPS)
    video.release();
    return fps


for allDir in pathDir:
    videopath = r'G:/1116_3/' + allDir   # 视频所在文件夹
    print(videopath)
    print(count(videopath))
    vc = cv2.VideoCapture(videopath)  # 读入视频文件

    c = 1

    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()

    else:
        rval = False

    timeF = 130  # 视频帧计数间隔频率

    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        if (c % timeF == 0):  # 每隔timeF帧进行存储操作
            cv2.imwrite(r'G:/1116_4/' + pathDir[i] +'_' +str(a) + '.jpg', frame)   # 存取图片文件夹

            a = a + 1
            # i = i + 1

        c = c + 1

        cv2.waitKey(1)
    a = 1
    i = i + 1
    vc.release()
print('视频按5秒存取已完成')
