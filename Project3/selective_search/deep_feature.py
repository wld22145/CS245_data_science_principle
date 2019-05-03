import pandas as pd
import os
import numpy as np
import time
import pickle
import selectivesearch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import warnings
import time
import time
from PIL import Image

start_time = time.time()

path = 'Animals_with_Attributes2/'

regions = np.load("ss_regions.npy")
print("regions loaded")

image_counter = 0


classname = pd.read_csv(path + 'classes.txt', header=None, sep='\t')
dic_class2name = {classname.index[i]: classname.loc[i][1] for i in range(classname.shape[0])}
dic_name2class = {classname.loc[i][1]: classname.index[i] for i in range(classname.shape[0])}

data_tf = transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs

model = models.resnet101(pretrained=True)

if torch.cuda.is_available():
    model = model.cuda()

model.eval()

exact_list = ['avgpool']
myexactor = FeatureExtractor(model, exact_list)


def resnet_feature(proposal):
    input_data = data_tf(proposal)
    input_data = torch.reshape(input_data,[1,3,224,224]).cuda()

    # print("input",input_data.size())
    feature_vector = myexactor(input_data)[0]
    feature_vector = torch.reshape(feature_vector,[1,2048])
    feature_vector = feature_vector.cpu().detach().numpy()
    return feature_vector

def deep_img(imagename):
    global image_counter
    # print("processing image No.", image_counter)
    img = Image.open(imagename).convert('RGB')
    # resize the image
    img = img.resize((224, 224))
    image_features = []
    for i in range(10):
        # print("region",regions[image_counter][i])
        (x, y, w, h) = regions[image_counter][i]
        if x == 0 and y ==0 and w ==0 and h==0:
            feature_vector = torch.zeros([2048]).numpy()
            image_features.append(feature_vector)
        else:
            img_arr = np.asarray(img,np.uint8)
            proposal = img_arr[x:x+w,y:y+h,:]
            proposal = np.uint8(proposal)
            proposal = Image.fromarray(proposal).convert('RGB')
            feature_vector = resnet_feature(proposal)
            # print("f v",feature_vector.shape)
            image_features.append(feature_vector)
    image_features = np.vstack(image_features)
    image_features = image_features.reshape(1,10,2048)
    # print("image features",image_features.shape)
    image_counter += 1


    return image_features


def deep_dir(imgDir, read_num='max'):
    dir_start_time = time.time()
    dir_features = []
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
        image_features = deep_img(imagename)
        dir_features.append(image_features)

    dir_features = np.vstack(dir_features)
    print("dir features",dir_features.shape)
    dir_finish_time = time.time()
    print("time consumption",dir_finish_time-dir_start_time)
    return dir_features


def deep_all(num):
    read_num = num
    features = []
    for i in range(50):
        item = dic_class2name[i]
        print("item No.",i,"name",item)
        dir_features = deep_dir(path + 'JPEGImages/' + item, read_num=read_num)
        features.append(dir_features)
    features = np.vstack(features)
    print("deep feature shape",features.shape)

    np.save("ss_resnet_features.npy",features)

    return features

features = deep_all(num='max')

finish_time = time.time()
print("time consumption",finish_time-start_time)