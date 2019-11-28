import glob
import os.path
import cv2
my_dir = r'H:/test3/'
out_dir = 'H:/test4/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
os.chdir(my_dir)
for name in glob.glob('*.jpg'):
    try:
        img = cv2.imread(name)
        cv2.imwrite(out_dir + name, img)
        print(name + 'load finish')
    except:
        print(name + 'picture is bad')
        break