#更改Labels中类别号
from glob2 import glob
import os, os.path, shutil
os.chdir('H:/data/Select/gongzhuang_anquanmao/labels1/')
files = glob('*.txt')
for txt in files:
    f1 = open('H:/data/Select/gongzhuang_anquanmao/labels2/'+ txt, 'w')
    # jpg = jpg.replace('\\', '/')
    # q = 0
    with open('H:/data/Select/gongzhuang_anquanmao/labels1/' + txt)as f:  # 返回一个文件对象
            list=f.readlines()
            for line in list:
                a = line.split(' ')[0]
                print(a)
                if a == '0':
                    a1=3
                    x1 = line.split(' ')[1]
                    y1 = line.split(' ')[2]
                    x2 = line.split(' ')[3]
                    y2 = line.split(' ')[4]
                    f1.write(str(a1) + ' ' + str(x1) + " " + str(y1) + ' ' + str(x2) + ' ' + str(y2) )
                elif a == '1':
                    a2 = 4
                    x1 = line.split(' ')[1]
                    y1 = line.split(' ')[2]
                    x2 = line.split(' ')[3]
                    y2 = line.split(' ')[4]
                    f1.write(str(a2) + ' ' + str(x1) + " " + str(y1) + ' ' + str(x2) + ' ' + str(y2) )
                elif a == '2':
                    a2 = 5
                    x1 = line.split(' ')[1]
                    y1 = line.split(' ')[2]
                    x2 = line.split(' ')[3]
                    y2 = line.split(' ')[4]
                    f1.write(str(a2) + ' ' + str(x1) + " " + str(y1) + ' ' + str(x2) + ' ' + str(y2) )
                elif a == '3':
                    a2 = 6
                    x1 = line.split(' ')[1]
                    y1 = line.split(' ')[2]
                    x2 = line.split(' ')[3]
                    y2 = line.split(' ')[4]
                    f1.write(str(a2) + ' ' + str(x1) + " " + str(y1) + ' ' + str(x2) + ' ' + str(y2) )
                elif a == '4':
                    a2 = 7
                    x1 = line.split(' ')[1]
                    y1 = line.split(' ')[2]
                    x2 = line.split(' ')[3]
                    y2 = line.split(' ')[4]
                    f1.write(str(a2) + ' ' + str(x1) + " " + str(y1) + ' ' + str(x2) + ' ' + str(y2) )
                elif a == '5':
                    a2 = 8
                    x1 = line.split(' ')[1]
                    y1 = line.split(' ')[2]
                    x2 = line.split(' ')[3]
                    y2 = line.split(' ')[4]
                    f1.write(str(a2) + ' ' + str(x1) + " " + str(y1) + ' ' + str(x2) + ' ' + str(y2) )
                elif a == '6':
                    a2 = 9
                    x1 = line.split(' ')[1]
                    y1 = line.split(' ')[2]
                    x2 = line.split(' ')[3]
                    y2 = line.split(' ')[4]
                    f1.write(str(a2) + ' ' + str(x1) + " " + str(y1) + ' ' + str(x2) + ' ' + str(y2) )
                else:
                    f1.write(line)
    #
    f1.close()


# print('标注文本平移翻转已完成，保存在1中')