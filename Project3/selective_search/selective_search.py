import pandas as pd
import os
import numpy as np
import time
import pickle
from skimage import io,transform
import selectivesearch

# NOTICE: opencv version must be correct
# pip install opencv-python==3.4.2.16
# pip install opencv-contrib-python==3.4.2.16

start_time = time.time()
path = 'Animals_with_Attributes2/'


classname = pd.read_csv(path + 'classes.txt', header=None, sep='\t')
dic_class2name = {classname.index[i]: classname.loc[i][1] for i in range(classname.shape[0])}
dic_name2class = {classname.loc[i][1]: classname.index[i] for i in range(classname.shape[0])}


def ss_img(imagename):
    img = io.imread(imagename)
    # resize the image
    img = transform.resize(img, (224, 224))
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)
    return img_lbl,regions


def ss_dir(imgDir, read_num='max'):
    dir_start_time = time.time()
    lbl_result = []
    reg_result = []
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
        img_lbl,regions = ss_img(imagename)
        lbl_result.append(img_lbl)
        reg_result.append(regions)
    dir_finish_time = time.time()
    print("time consumption",dir_finish_time-dir_start_time)
    return lbl_result,reg_result


def ss_all(num):
    read_num = num
    lbl_result = []
    reg_result = []
    for i in range(50):
        item = dic_class2name[i]
        print("item No.",i,"name",item)
        dir_lbl_result,dir_reg_result = ss_dir(path + 'JPEGImages/' + item, read_num=read_num)
        lbl_result.extend(dir_lbl_result)
        reg_result.extend(dir_reg_result)
    print("image label result size",len(lbl_result))
    print("region result size", len(reg_result))
    return lbl_result,reg_result

lbl_result,reg_result = ss_all(num='max')

print("Saving pickle file")
pickle.dump(lbl_result,"lbl_result.pkl")
pickle.dump(reg_result,"reg_result.pkl")

finish_time = time.time()
print("Total Time Consumption",finish_time-start_time)