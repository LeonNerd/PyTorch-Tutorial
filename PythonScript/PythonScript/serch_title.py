#提起txt文件中的title
f1 = open('H:/data/test1.txt', 'w')
with open('H:/data/test.txt')as f:  # 返回一个文件对象
        list=f.readlines()
        for line in list:
            a = line.split('.')[0]
            b = a.split('smsjdata/')[1]
            print(b)
            f1.write(b + '\n')
f1.close()
