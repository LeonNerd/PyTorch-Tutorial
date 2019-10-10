import os
#批量删除指定文件夹下的指定文件


#定义一个返回所有图片绝对路径的函数
def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result

def main():
    path = 'H:/VOC07_12/'
    list1 = all_path(path)

    remove_path = 'H:/de.txt'
    with open(remove_path) as f:
        list2 = list(map(lambda s:s.strip(), f.readlines()))

#得到所有图片的名字并添加到list3中
    list3 = []
    for i in range(len(list1)):
        line = os.path.split(list1[i])[-1].split('/')[0]
        fname = os.path.splitext(line)[0]
        list3.append(fname)

#将需要删除的图片的路径添加到list4中
    list4 = []
    for j in range(len(list3)):
        for k in range(len(list2)):
            if list3[j] == list2[k]:
                out_path = list1[j]
                list4.append(out_path)

    for n in range(len(list4)):
        os.remove(list4[n])

if __name__ == '__main__':
    main()
