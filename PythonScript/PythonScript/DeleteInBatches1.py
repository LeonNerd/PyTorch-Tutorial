#剔除Labels中无用类别号的文件
from glob2 import glob
import os, os.path, shutil
os.chdir('H:/data/luneng_security/labels/')
files = glob('*.txt')
for fileneme in files:
    flag = 0
    f1 = open('H:/data/luneng_security/labels2/'+ fileneme, 'w')
    with open('H:/data/luneng_security/labels/' + fileneme)as f:  # 返回一个文件对象
        list=f.readlines()
        b = 6
        for line in list:
            a = line.split(' ')[0]
            a = int(a)
            if a <= b:
                f1.write(line)
                flag = 1
    f1.close()
    if flag == 0:
        os.remove('H:/data/luneng_security/labels2/'+ fileneme)
