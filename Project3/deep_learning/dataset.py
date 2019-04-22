import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image
import time

start_time = time.time()
image_size = 224
path = 'Animals_with_Attributes2/'

classname = pd.read_csv(path + 'classes.txt', header=None, sep='\t')
dic_class2name = {classname.index[i]: classname.loc[i][1] for i in range(classname.shape[0])}
dic_name2class = {classname.loc[i][1]: classname.index[i] for i in range(classname.shape[0])}


def load_Img(imgDir, read_num='max'):
    imgs = os.listdir(imgDir)
    imgs = np.ravel(pd.DataFrame(imgs).sort_values(by=0).values)
    if read_num == 'max':
        imgNum = len(imgs)
    else:
        imgNum = read_num
    data = np.empty((imgNum, image_size, image_size, 3), dtype="float32")
    print(imgNum)
    for i in range(imgNum):
        if i%100==0:
            print("processing Image No.",i)
        img = Image.open(imgDir + "/" + imgs[i])
        arr = np.asarray(img, dtype="float32")
        if arr.shape[1] > arr.shape[0]:
            arr = cv2.copyMakeBorder(arr, int((arr.shape[1] - arr.shape[0]) / 2),
                                     int((arr.shape[1] - arr.shape[0]) / 2), 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            arr = cv2.copyMakeBorder(arr, 0, 0, int((arr.shape[0] - arr.shape[1]) / 2),
                                     int((arr.shape[0] - arr.shape[1]) / 2), cv2.BORDER_CONSTANT,
                                     value=0)
        arr = cv2.resize(arr, (image_size, image_size))
        if len(arr.shape) == 2:
            temp = np.empty((image_size, image_size, 3))
            temp[:, :, 0] = arr
            temp[:, :, 1] = arr
            temp[:, :, 2] = arr
            arr = temp
        data[i, :, :, :] = arr
    return data, imgNum


def load_data(train_classes, test_classes, num):
    read_num = num

    traindata_list = []
    trainlabel_list = []
    testdata_list = []
    testlabel_list = []

    for item in train_classes.iloc[:, 0].values.tolist():
        tup = load_Img(path + 'JPEGImages/' + item, read_num=read_num)
        traindata_list.append(tup[0])
        trainlabel_list += [dic_name2class[item]] * tup[1]

    for item in test_classes.iloc[:, 0].values.tolist():
        tup = load_Img(path + 'JPEGImages/' + item, read_num=read_num)
        testdata_list.append(tup[0])
        testlabel_list += [dic_name2class[item]] * tup[1]

    return np.row_stack(traindata_list), np.array(trainlabel_list), np.row_stack(testdata_list), np.array(
        testlabel_list)


train_classes = pd.read_csv(path + 'trainclasses.txt', header=None)
test_classes = pd.read_csv(path + 'testclasses.txt', header=None)

traindata, trainlabel, testdata, testlabel = load_data(train_classes, test_classes, num='max')

print(traindata.shape, trainlabel.shape, testdata.shape, testlabel.shape)

data = np.vstack((traindata,testdata))
label = np.concatenate((trainlabel,testlabel))
print(data.shape,label.shape)

print("Saving npy file")
np.save(path + 'AWA2_224_traindata.npy', traindata)
np.save(path + 'AWA2_224_testdata.npy', testdata)

np.save(path + 'AWA2_trainlabel.npy', trainlabel)
np.save(path + 'AWA2_testlabel.npy', testlabel)

np.save(path + 'AWA2_data.npy', data)
np.save(path + 'AWA2_label.npy', label)

finish_time = time.time()
print("Time consumption",finish_time-start_time)