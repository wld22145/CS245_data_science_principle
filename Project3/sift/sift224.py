# images are resized to (224,224) in this program
import pandas as pd
import os
import numpy as np
import cv2
import time
import pickle

# NOTICE: opencv version must be correct
# pip install opencv-python==3.4.2.16
# pip install opencv-contrib-python==3.4.2.16

start_time = time.time()
path = 'Animals_with_Attributes2/'


classname = pd.read_csv(path + 'classes.txt', header=None, sep='\t')
dic_class2name = {classname.index[i]: classname.loc[i][1] for i in range(classname.shape[0])}
dic_name2class = {classname.loc[i][1]: classname.index[i] for i in range(classname.shape[0])}

def convert_kp(kp):
    result = []
    for point in kp:
        # data structure of key point
        tmp = [point.pt, point.size, point.angle, point.response, point.octave, point.class_id]
        result.append(tmp)
    return result

def sift_img(imagename):
    image = cv2.imread(imagename)
    image = cv2.resize(image,(224,224))
    sift = cv2.xfeatures2d_SIFT.create()
    kp, des = sift.detectAndCompute(image, None)
    kp = convert_kp(kp)
    return kp,des


def sift_dir(imgDir, read_num='max'):
    dir_start_time = time.time()
    kp_result = []
    des_result = []
    imgs = os.listdir(imgDir)
    imgs = np.ravel(pd.DataFrame(imgs).sort_values(by=0).values)
    if read_num == 'max':
        imgNum = len(imgs)
    else:
        imgNum = read_num
    print("Number of Images:",imgNum)
    for i in range(imgNum):
        if i%100==0:
            print("processing Image No.",i)
        imagename = imgDir + "/" + imgs[i]
        kp,des = sift_img(imagename)
        kp_result.append(kp)
        des_result.append(des)
    dir_finish_time = time.time()
    print("time consumption",dir_finish_time-dir_start_time)
    return kp_result,des_result


def sift_all(num):
    read_num = num
    kp_result = []
    des_result = []
    for i in range(50):
        item = dic_class2name[i]
        print("item No.", i, "name", item)
        dir_kp_result,dir_des_result = sift_dir(path + 'JPEGImages/' + item, read_num=read_num)
        kp_result.extend(dir_kp_result)
        des_result.extend(dir_des_result)
    print("key point result size",len(kp_result))
    print("descriptor result size", len(des_result))
    return kp_result,des_result

kp_result,des_result = sift_all(num='max')

print("Saving pickle file")
pickle.dump(kp_result,open("kp_result224.pkl","wb"))
pickle.dump(des_result,open("des_result224.pkl","wb"))

finish_time = time.time()
print("Total Time Consumption",finish_time-start_time)