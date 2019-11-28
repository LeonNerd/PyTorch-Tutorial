# encoding=utf-8
import multiprocessing as mp
import cv2
import time
from imutils.video import FPS

import threading
import queue
'''2018-05-21 Yonv1943'''
'''2018-07-02 setattr(), run_multi_camera()'''


def queue_img_put(q, name, pwd, ip, channel=1):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d?tcp" % (name, pwd, ip, channel))
    time.sleep(1)

    while True:
        is_opened, frame = cap.read()
        q.put(frame) if is_opened else None
        q.get() if q.qsize() > 1 else None


def queue_img_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


def run():  # single camera
    user_name, user_pwd, camera_ip = "admin", "nowvciji121226", "192.168.1.64"

    Q = queue.Queue(maxsize=2)  # 构造一个不限制大小的的队列
    _WORKER_THREAD_NUM = 2  # 设置线程的个数

    processes = [threading.Thread(target=queue_img_put, args=(Q, user_name, user_pwd, camera_ip)),
                 threading.Thread(target=queue_img_get, args=(Q, camera_ip))]

    [setattr(process, "daemon", True) for process in processes]  # process.daemon = True
    [process.start() for process in processes]
    [process.join() for process in processes]




def run_multi_camera():
    user_name, user_pwd = "admin", "huarui2019."
    user_name3, user_pwd3, camera_ip3 = "admin", "nowvciji121226", "192.168.1.64"
    camera_ip_l = [
        "192.168.1.44",
        "192.168.1.43",
    ]

    mp.set_start_method(method='spawn')  # init

    queues = [mp.Queue(maxsize=2) for _ in camera_ip_l]

    processes = []
    for queue, camera_ip in zip(queues, camera_ip_l):
        processes.append(mp.Process(target=queue_img_put, args=(queue, user_name, user_pwd, camera_ip)))
        processes.append(mp.Process(target=queue_img_get, args=(queue, camera_ip)))

    processes.append(mp.Process(target=queue_img_put, args=(queue, user_name3, user_pwd3, camera_ip3)))
    processes.append(mp.Process(target=queue_img_get, args=(queue, camera_ip3)))	
    [setattr(process, "daemon", True) for process in processes]  # process.daemon = True
    [process.start() for process in processes]
    [process.join() for process in processes]


if __name__ == '__main__':
    run()
    #run_multi_camera()
