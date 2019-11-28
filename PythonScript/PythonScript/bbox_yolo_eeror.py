# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:55:43 2015

This script is to convert the txt annotation files to appropriate format needed by YOLO

"""

import os
from os import walk, getcwd
from PIL import Image
import shutil
from glob2 import glob

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


"""-------------------------------------------------------------------"""

""" Configure Paths"""
img_folder = "H:/data/cover/fz1/"
mypath = "H:/data/cover/txt_flip/"
outpath = "H:/data/cover/txt_flipY/"
errorpath = "H:/data/cover/error1/"
if not os.path.exists(outpath):
    os.makedirs(outpath)
if not os.path.exists(errorpath):
    os.makedirs(errorpath)
# class_map = {0: 10, 1: 11, 2: 10, 3: 11, 4: 12}

""" Get input text file list """
img_name_list = []
for (dirpath, dirnames, filenames) in walk(img_folder):
    img_name_list.extend(filenames)
    break
# os.chdir(img_folder)
# filenames = glob('*.jpg')
# img_name_list.extend(filenames)
# # print(txt_name_list)
# print(img_name_list)


""" Process """
for img_name in img_name_list:
    if (img_name[0] != '.'):

        """ Open input text files """
        txt_path = mypath + os.path.splitext(img_name)[0] + '.txt'
        error_txt_path = errorpath + os.path.splitext(img_name)[0] + '.txt'
        print("Input:" + txt_path)
        txt_file = open(txt_path, "r")
        lines = txt_file.read().split('\n')  # for ubuntu, use "\r\n" instead of "\n"

        """ Open output text files """
        txt_outpath = outpath + os.path.splitext(img_name)[0] + '.txt'
        print("Output:" + txt_outpath)
        txt_outfile = open(txt_outpath, "w")

        """ Convert the data to YOLO format """
        flag = 0
        ct = 0
        for line in lines:
            if (len(line) >= 4):
                ct = ct + 1
                # print(line)
                elems = line.split(' ')
                # print(elems)
                cls_id = int(elems[0])
                xmin = int(float(elems[1]))
                xmax = int(float(elems[3]))
                ymin = int(float(elems[2]))
                ymax = int(float(elems[4]))

                img_path = str('%s%s' % (img_folder, img_name))
                error_img_path = errorpath + os.path.splitext(img_name)[0] + '.jpg'
                im = Image.open(img_path)
                w = int(im.size[0])
                h = int(im.size[1])

                xmin = max(1, xmin)
                ymin = max(1, ymin)
                xmax = min(xmax, w - 1)
                ymax = min(ymax, h - 1)

                # print(w, h)
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = convert((w, h), b)
                if (bb[2] < 0 or bb[3] < 0):
                    flag = 1
                # print(bb)
                txt_outfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                # txt_outfile.write(str(0) + " " + " ".join([str(a) for a in bb]) + '\n')
                # print('')
        if (flag == 1):
            print("Error, w or h is negative=================================================")
            shutil.copy(txt_path, error_txt_path)
            shutil.copy(img_path, error_img_path)
            # break

        # """ Save those images with bb into list"""
        # if(ct != 0):
        #     list_file.write('%s/images/%s/%s.JPEG\n'%(wd, cls, os.path.splitext(txt_name)[0]))

# list_file.close()

