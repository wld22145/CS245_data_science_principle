import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Subset
from torchvision import transforms
import copy
import random
import numpy as np
import os

TRAIN_DATA_RATIO = 0.9
root_path = "OfficeHomeDataset_10072016/"
art_path = root_path + "Art"
clipart_path = root_path + "Clipart"
product_path = root_path + "Product"
real_path = root_path + "Real World"

transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def art_data_loader():
    data = ImageFolder(art_path, transform)
    if not (os.path.exists("art_train_indices.npy") and os.path.exists("art_test_indices.npy")):
        random.seed(0)
        indices = [i for i in range(len(data))]
        random.shuffle(indices)
        train_indices = indices[:int(TRAIN_DATA_RATIO * len(indices))]
        test_indices = indices[int(TRAIN_DATA_RATIO * len(indices)):]
        np.save("art_train_indices.npy", np.array(train_indices))
        np.save("art_test_indices.npy", np.array(test_indices))

    train_indices = np.load("art_train_indices.npy")
    test_indices = np.load("art_test_indices.npy")
    print("art train data", len(train_indices))
    print("art test data", len(test_indices))

    train_dataset = Subset(data, train_indices)
    test_dataset = Subset(data, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def clipart_data_loader():
    data = ImageFolder(clipart_path, transform)
    if not (os.path.exists("clipart_train_indices.npy") and os.path.exists("clipart_test_indices.npy")):
        random.seed(0)
        indices = [i for i in range(len(data))]
        random.shuffle(indices)
        train_indices = indices[:int(TRAIN_DATA_RATIO * len(indices))]
        test_indices = indices[int(TRAIN_DATA_RATIO * len(indices)):]
        np.save("clipart_train_indices.npy", np.array(train_indices))
        np.save("clipart_test_indices.npy", np.array(test_indices))

    train_indices = np.load("clipart_train_indices.npy")
    test_indices = np.load("clipart_test_indices.npy")
    print("clipart train data", len(train_indices))
    print("clipart test data", len(test_indices))

    train_dataset = Subset(data, train_indices)
    test_dataset = Subset(data, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def product_data_loader():
    data = ImageFolder(product_path, transform)
    if not (os.path.exists("product_train_indices.npy") and os.path.exists("product_test_indices.npy")):
        random.seed(0)
        indices = [i for i in range(len(data))]
        random.shuffle(indices)
        train_indices = indices[:int(TRAIN_DATA_RATIO * len(indices))]
        test_indices = indices[int(TRAIN_DATA_RATIO * len(indices)):]
        np.save("product_train_indices.npy", np.array(train_indices))
        np.save("product_test_indices.npy", np.array(test_indices))

    train_indices = np.load("product_train_indices.npy")
    test_indices = np.load("product_test_indices.npy")
    print("product train data", len(train_indices))
    print("product test data", len(test_indices))

    train_dataset = Subset(data, train_indices)
    test_dataset = Subset(data, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def real_data_loader():
    data = ImageFolder(real_path, transform)
    if not (os.path.exists("real_train_indices.npy") and os.path.exists("real_test_indices.npy")):
        random.seed(0)
        indices = [i for i in range(len(data))]
        random.shuffle(indices)
        train_indices = indices[:int(TRAIN_DATA_RATIO * len(indices))]
        test_indices = indices[int(TRAIN_DATA_RATIO * len(indices)):]
        np.save("real_train_indices.npy", np.array(train_indices))
        np.save("real_test_indices.npy", np.array(test_indices))

    train_indices = np.load("real_train_indices.npy")
    test_indices = np.load("real_test_indices.npy")
    print("real train data", len(train_indices))
    print("real test data", len(test_indices))

    train_dataset = Subset(data, train_indices)
    test_dataset = Subset(data, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def adaptation_data_loader():
    art_data = ImageFolder(art_path, transform)
    art_loader = DataLoader(art_data, batch_size=64, shuffle=True)
    clipart_data = ImageFolder(clipart_path, transform)
    clipart_loader = DataLoader(clipart_data, batch_size=64, shuffle=True)
    product_data = ImageFolder(product_path, transform)
    product_loader = DataLoader(product_data, batch_size=64, shuffle=True)
    real_data = ImageFolder(real_path, transform)
    real_loader = DataLoader(real_data, batch_size=64, shuffle=False)
    return art_loader, clipart_loader, product_loader, real_loader


if __name__ == "__main__":
    art_train_loader, art_test_loader = art_data_loader()
    clipart_train_loader, clipart_test_loader = clipart_data_loader()
    product_train_loader, product_test_loader = product_data_loader()
    real_train_loader, real_test_loader = real_data_loader()
    art_loader, clipart_loader, product_loader, real_loader  = adaptation_data_loader()
