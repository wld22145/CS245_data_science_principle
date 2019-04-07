import time
import numpy as np
import os
from config import path_dir
from sklearn.model_selection import KFold
import random

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
        features_np.dump("features_np.npy")
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

def prepare_dataset():
    if not os.path.exists("features_shuffled.npy"):
        features = parse_features()
        labels = parse_labels()
        random.seed(time.time())
        shuffle_list = [i for i in range(data_size)]
        random.shuffle(shuffle_list)
        features_shuffled = []
        labels_shuffled = []
        for i in range(data_size):
            idx = shuffle_list[i]
            features_shuffled.append(features[idx])
            labels_shuffled.append(labels[idx])
        features_shuffled = np.array(features_shuffled)
        labels_shuffled = np.array(labels_shuffled)
        print("saving shuffled dataset")
        features_shuffled.dump("features_shuffled.npy")
        labels_shuffled.dump("labels_shuffled.npy")
    features = np.load("features_shuffled.npy")
    labels = np.load("labels_shuffled.npy")
    return features,labels

def shuffle_dataset(features,labels):
    random.seed(time.time())
    shuffle_list = [i for i in range(data_size)]
    random.shuffle(shuffle_list)
    features_shuffled = []
    labels_shuffled = []
    for i in range(data_size):
        idx = shuffle_list[i]
        features_shuffled.append(features[idx])
        labels_shuffled.append(labels[idx])
    features_shuffled = np.array(features_shuffled)
    labels_shuffled = np.array(labels_shuffled)
    return features_shuffled,labels_shuffled


# prepare data
filenames = parse_filenames()
features,labels = prepare_dataset()

show_information(0)


# TODO: dimension reduction
# here I simply choose the first 50 features as example
reduced_features = features[:,:50]
print("reduced features shape:",reduced_features.shape)

total_train_features = reduced_features[:int(data_size*0.6)]
total_train_labels = labels[:int(data_size*0.6)]
test_features = reduced_features[int(data_size*0.6):]
test_labels = labels[int(data_size*0.6):]
print("train features shape:",total_train_features.shape)
print("train labels shape:",total_train_labels.shape)
print("test features shape:",test_features.shape)
print("test labels shape:",test_labels.shape)
# k-fold validation
kfold= KFold(n_splits=5,random_state =None)
fold_cnt = 0
for train_index,valid_index in kfold.split(total_train_features,total_train_labels):
    train_features = total_train_features[train_index]
    train_labels = total_train_labels[train_index]
    valid_features = total_train_features[valid_index]
    valid_labels = total_train_labels[valid_index]
    print("test")
    # print(train_features[0])
    print(train_labels[0])
    # print(valid_features[0])
    print(valid_labels[0])


    fold_cnt += 1
    print("fold:", fold_cnt)
    print("train features shape:",train_features.shape)
    print("train labels shape:",train_labels.shape)
    print("valid features shape:",valid_features.shape)
    print("valid labels shape:", valid_labels.shape)
    # TODO: machine learning algorithm
    # based on train/valid features/labels, you can conduct further experiments
    #


# finally, examine the model with tran/test features

finish_time = time.time()
print("time consumption",finish_time-start_time)