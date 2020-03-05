import glob, os
import cv2

path_data        = r'F:\0214\pict_jpg/'


for pathAndFilename in glob.iglob(os.path.join(path_data, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    print(title)

    img = cv2.imread(pathAndFilename)
    img_gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(pathAndFilename, img_color)
    
