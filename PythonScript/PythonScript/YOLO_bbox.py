# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:55:43 2015

This script is to convert the txt annotation files to appropriate format needed by YOLO 

"""

import os
import math
from os import walk, getcwd
from PIL import Image
from glob2 import glob

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def reverseConvert(size, box):
    tw = size[0]
    th = size[1]
    x = tw*box[0]
    y = th*box[1]
    w = tw*box[2]
    h = th*box[3]
    l = x - w/2.0
    r = x + w/2.0
    t = y - h/2.0
    b = y + h/2.0
    return (int(math.ceil(l)),int(math.ceil(t)),int(math.floor(r)),int(math.floor(b)))

    
    
"""-------------------------------------------------------------------""" 

""" Configure Paths"""   
img_folder = "H:/data/cover/img/"
mypath     = "H:/data/cover/labels/"
outpath    = "H:/data/cover/bbox/"
if not os.path.exists(outpath ):
    os.makedirs(outpath)

""" Get input text file list """
txt_name_list = []
# os.chdir(mypath)
# filenames = glob('*.txt')
# txt_name_list.extend(filenames)
# print(txt_name_list)
for (dirpath, dirnames, filenames) in walk(mypath):
    txt_name_list.extend(filenames)
    break
print(txt_name_list)

# class_map = {0 : 3, 1 : 3, 2 : 4, 3 : 2}

""" Process """
for txt_name in txt_name_list:
    if (txt_name[0] != '.'):
        img_path = str('%s%s.jpg'%(img_folder, os.path.splitext(txt_name)[0]))
        if not os.path.isfile(img_path):
            continue
    
        """ Open input text files """
        txt_path = mypath + txt_name
        print("Input:" + txt_path)
        txt_file = open(txt_path, "r")
        ct = 0
        while True:
            line = txt_file.readline()
            if not line:
                break
            ct = ct + 1
        txt_file.close()

        """ Open output text files """
        txt_outpath = outpath + txt_name
        print("Output:" + txt_outpath)
        txt_outfile = open(txt_outpath, "w")
        txt_outfile.write(str(ct) + '\n')


        """ Reverse convert from YOLO to BBox format """
        txt_file = open(txt_path, "r")
        while True:
            line = txt_file.readline()
            if not line:
                break
            elems = line.split(' ')
            print(elems)
            cls_id = int(elems[0])
            b = (float(elems[1]), float(elems[2]), float(elems[3]), float(elems[4]))

            img_path = str('%s%s.jpg'%(img_folder, os.path.splitext(txt_name)[0]))
            im=Image.open(img_path)
            w= int(im.size[0])
            h= int(im.size[1])

            bb = reverseConvert((w,h), b)
            print(bb)
            txt_outfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            print('')
        txt_file.close()

        
# list_file.close()       
