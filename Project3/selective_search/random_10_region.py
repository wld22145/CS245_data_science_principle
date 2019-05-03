import pandas as pd
import os
import numpy as np
import time
import pickle
from skimage import io,transform
import selectivesearch
import time
import matplotlib.pyplot as plt
import random

start_time = time.time()

path = 'Animals_with_Attributes2/'

print("loading file")
reg_result = pickle.load(open("reg_result.pkl", "rb"))
print("file loaded")


poor_count = 0
def get_size(element):
    return element["size"]

regions = []
for i in range(len(reg_result)):
    img_reg = []
    for j in range(len(reg_result[i])):
        if reg_result[i][j]["size"] > 1000 and reg_result[i][j]["size"] < 10000:
            img_reg.append(reg_result[i][j])
    if len(img_reg) >= 10:
        img_reg_10 = random.sample(img_reg,10)
        img_reg_result = []
        for j in range(10):
            img_reg_result.append([img_reg_10[j]["rect"]])
        regions.append(img_reg_result)
    else:
        img_reg_result = []
        for j in range(len(img_reg)):
            img_reg_result.append([img_reg[j]["rect"]])
        for j in range(10-len(img_reg)):
            img_reg_result.append([(0,0,0,0)])
        regions.append(img_reg_result)


regions = np.array(regions)
regions = regions.squeeze()
print(regions.shape)
print(regions)

np.save("ss_regions.npy",regions)

finish_time = time.time()
print("time consumption",finish_time-start_time)