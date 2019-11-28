# -*- coding: utf-8 -*-
import shutil


def objFileName():
    local_file_name_list = "H:/data/copy.txt"
    obj_name_list = []
    for i in open(local_file_name_list, 'r'):
        obj_name_list.append(i.replace('\n', '') + '.txt')

    return obj_name_list


def copy_img():
    local_img_name = r'H:/data/1/'
    # 指定要复制的图片路径
    path = r'H:/data/3/'
    # 指定存放图片的目录
    for i in objFileName():
        new_obj_name = i
        shutil.copy(local_img_name + '/' + new_obj_name, path + '/' + new_obj_name)


if __name__ == '__main__':
    copy_img()
