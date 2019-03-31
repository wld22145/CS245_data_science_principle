import time
import numpy as np
import os
from config import path_dir
from sklearn.model_selection import KFold

path_feature = path_dir + r"\AwA2-features.txt"
path_filename = path_dir + r"\AwA2-filenames.txt"
path_label = path_dir + r"\AwA2-labels.txt"

data_size = 37322
feature_size = 2048

start_time = time.time()

def parse_labels():
    file_labels = open(path_label)
    labels = file_labels.readlines()
    for i in range(data_size):
        labels[i] = int(labels[i][:-1])
    labels = np.array(labels)
    print("labels shape:",labels.shape)
    return labels

def parse_filenames():
    file_filenames = open(path_filename)
    filenames = file_filenames.readlines()
    for i in range(data_size):
        filenames[i] = filenames[i][:-1]
    return filenames

def parse_features():
    file_features = open(path_feature)
    if not os.path.exists("features_np.npy"):
        print("parsing features")
        features = file_features.readlines()
        for i in range(data_size):
            if i%100==0:
                print("parsing progress:",i)
            features[i] = features[i].split()
            for j in range(feature_size):
                features[i][j] = float(features[i][j])
        print(len(features[0]))
        features_np = np.array(features)
        features_np.dump()
        print("features saved as features_np.npy")
    features = np.load("features_np.npy")
    print("features loaded from features_np.npy")
    print("features shape:",features.shape)
    return features

def show_information(index):
    print("index",index)
    print("label:",labels[index])
    print("filename:",filenames[index])
    print("feature:",features[index])

# prepare data
filenames = parse_filenames()
labels = parse_labels()
features = parse_features()

show_information(0)

# TODO: dimension reduction
# here I simply choose the first 50 features as example
reduced_features = features[:,:50]
print("reduced features shape:",reduced_features.shape)

# k-fold validation
kfold= KFold(n_splits=5,random_state =None)
fold_cnt = 0
for train_index,test_index in kfold.split(reduced_features,labels):
    train_features = reduced_features[train_index]
    train_labels = labels[train_index]
    test_features = reduced_features[test_index]
    test_labels = labels[test_index]
    fold_cnt += 1
    print("fold:", fold_cnt)
    print("train features shape:",train_features.shape)
    print("train labels shape:",train_labels.shape)
    print("test features shape:",test_features.shape)
    print("test labels shape:", test_labels.shape)
    # TODO: machine learning algorithm
    # based on train/test features/labels, you can conduct further experiments
    #


finish_time = time.time()
print("time consumption",finish_time-start_time)