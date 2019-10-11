# -*- coding: utf-8 -*-
'''
将标注的xml文件转成bbox文件
'''
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

cls_map = {'TaDiao':4, 'ShiGongJiXie':2, 'DiaoChe':3, 'YanHuo':5}

def xml_to_bbox(xml_file, output_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find('filename').text
    if filename[-4:] !='.jpg':
        filename = filename + '.jpg'
    print (filename)
    width = int(root.find('size')[0].text)
    height = int(root.find('size')[1].text)
    if width==0 or height ==0:
        return
    cnt = len(root.findall('object'))
    bbox_file = os.path.join(output_path, os.path.splitext(filename)[0]+'.txt' )
    txt_outfile = open(bbox_file, "w")
    txt_outfile.write(str(cnt) + '\n')

    for member in root.findall('object'):
        cls_test = member[0].text
        coordinate = member.find('bndbox')
        xmin = coordinate[0].text
        ymin = coordinate[1].text
        xmax = coordinate[2].text
        ymax = coordinate[3].text
        bb = (xmin, ymin, xmax, ymax)
        print (cls_test)
        print (bb)
        if cls_test not in cls_map.keys():
            continue
        txt_outfile.write(str(cls_map[cls_test]) + " " + " ".join([str(a) for a in bb]) + '\n')
    txt_outfile.close()
    print ("")



def main():
    xml_path = '/Users/lilong/Documents/Huarui/labeling/BBox-Label-Tool/Images/labels_xml/'
    output_path = '/Users/lilong/Documents/Huarui/labeling/BBox-Label-Tool/Images/labels_bbox/'
    for xml_file in glob.iglob(os.path.join(xml_path, '*.xml')):
        xml_df = xml_to_bbox(xml_file, output_path)
    print('Successfully converted xml to csv.')

main()