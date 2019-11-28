#图片画框检测Labels_YOLO质量
import os
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.patches as patches

label_dict = {'0':'people','2':'no_helmet','1':'normal'}

images_dir = "H:/data/Select/tuziyolo"
labels_dir = 'H:/data/Select/1'

labels_dir_list = os.listdir(labels_dir)
for x in labels_dir_list:
	try:
		labels_file = labels_dir + "/" + x  #TXT文件的路径信息

		a = x.split(".")[0]
		b = x.split(".")[1]
		# print(a)

		#filename = x.split(".")[0]+"."+x.split(".")[1]+ ".jpg"  #生成的图片名称
		filename = x.split(".")[0] + "." + "jpg"  # 生成的图片名称
		img_filename = images_dir + "/" + filename  # 图片路径信息

		img = Image.open(img_filename)   # 图片格式 宽高等信息

		img_size = img.size
		xx = img_size[0]  # 宽
		yy = img_size[1]  # 高
		plt.imshow(img)
		plt.axis('off')
		# plt.show()

		file = open(labels_file,"r",encoding = "UTF-8")
		for line in file.readlines():    #按行读取文件信息
			line = line.strip("\n")      #字符串类型 strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
			new_line = line.split(" ")    #列表类型  ['0', '0.421680', '0.485069', '0.024609', '0.168750']

			# if len(new_line) == 1:
			# 	continue
			# else:
			lbtype = label_dict[new_line[0]]   #读出的是什么类型 ，人/正常/异常
			x1 = float(new_line[1]) * xx       #图片中心点X坐标
			y1 = float(new_line[2]) * yy
			x2 = float(new_line[3]) * xx	   #标注框宽
			y2 = float(new_line[4]) * yy
			print(x1,x2,y1,y2)
			rect = patches.Rectangle((x1-0.5*x2, y1-0.5*y2), x2, y2,linewidth = 0.5,color='#00FF00', fill=False)
			plt.gca().add_patch(rect)
			plt.text(x1, y1+y2*0.5, new_line[0] , size = 6, alpha = 1,color = "red")

		plt.savefig("H:/data/Select/plot/" + filename,dpi=400)
		print("labeling:",x)
		plt.close()

	except Exception as e:
		print(e)

