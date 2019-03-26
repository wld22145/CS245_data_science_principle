import time
import numpy as np
import os

path_dir = r"D:\Lab\awa2\AwA2-features\Animals_with_Attributes2\Features\ResNet101"
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

filenames = parse_filenames()
labels = parse_labels()
features = parse_features()

show_information(0)

finish_time = time.time()
print("time consumption",finish_time-start_time)