import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch import nn, optim
import warnings
import time

start_time = time.time()

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

path = 'Animals_with_Attributes2/'

classname = pd.read_csv(path + 'classes.txt', header=None, sep='\t')
dic_class2name = {classname.index[i]: classname.loc[i][1] for i in range(classname.shape[0])}
dic_name2class = {classname.loc[i][1]: classname.index[i] for i in range(classname.shape[0])}


class dataset(Dataset):
    def __init__(self, data, label, transform):
        super().__init__()
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.label[index]

    def __len__(self):
        return self.data.shape[0]


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


data = np.load(path + 'AWA2_data.npy')
label = np.load(path + 'AWA2_label.npy')

data_tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dataset = dataset(data, label, data_tf)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = models.resnet152(pretrained=True)

if torch.cuda.is_available():
    model = model.cuda()

model.eval()

exact_list = ['avgpool']
myexactor = FeatureExtractor(model, exact_list)

feature_list = []
for data in tqdm(loader):
    img, label = data
    if torch.cuda.is_available():
        with torch.no_grad():
            img = Variable(img).cuda()
        with torch.no_grad():
            label = Variable(label).cuda()
    else:
        with torch.no_grad():
            img = Variable(img)
        with torch.no_grad():
            label = Variable(label)
    feature = myexactor(img)[0]
    feature = feature.resize(feature.shape[0], feature.shape[1])
    feature_list.append(feature.detach().cpu().numpy())

features = np.row_stack(feature_list)

print("features shape",features.shape)
np.save('resnet152_features.npy', features)

finish_time = time.time()
print("Time consumption",finish_time-start_time)