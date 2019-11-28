import cv2
import codecs
import time
import math
import os
import sys
import codecs
from glob2 import glob

#扫描原文件夹
os.getcwd()
os.chdir('E:/123456/labels')
files = glob('*')
print(files)

#对应生成新的文件夹
os.chdir('E:/123456/labels1')
for txt in files:
    os.mkdir(txt)