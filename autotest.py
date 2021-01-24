#I attach it as a link because the data is too large.
#https://zenodo.org/record/3678171#.YA1hj8ht9PY

from scipy.io.wavfile import read
import os
from os import listdir
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from google.colab import drive #if you use colab, erase '#'

from torch.autograd import Variable as V
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import random
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

"""
random_seed = 777
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
"""
device = 'cpu'

resample_rate = 8000
resample_value = 80000


#train1 = os.listdir("train12/") 
test1 = os.listdir("test12/")

#train_data_set = []
test_data_set = []

#for i in train1:
#   tmp_train = librosa.load("train12/"+str(i), sr=160000)[0]
    # tmp_train = librosa.resample(tmp_train, orig_sr=16000, target_sr=2000)
#   train_data_set.append(tmp_train)

for l in test1:
    tmp_test = librosa.load("test12/"+str(l), sr=160000)[0]
    test_data_set.append(tmp_test)

#train_concate = np.concatenate(train_data_set, axis=0)
#train_concate = train_concate.reshape(-1, 160000)
#train_concate = np.expand_dims(train_concate, 1)

test_concate = np.concatenate(test_data_set, axis=0)
test_concate = test_concate.reshape(-1, 160000)
test_concate = np.expand_dims(test_concate, 1)

#x_train_data = DataLoader(tmp_train, batch_size=20000)
x_test_data = DataLoader(tmp_test, batch_size=20000)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(20000, 3000)
        self.hidden1 = nn.Linear(3000, 1000)
        self.hidden2 = nn.Linear(1000, 3000)
        self.decoder = nn.Linear(3000, 20000)

    def forward(self, data):
        encoded = self.encoder(data)
        hidden_o1 = torch.tanh(self.hidden1(encoded))
        hidden_o2 = torch.tanh(self.hidden2(hidden_o1))
        output = self.decoder(hidden_o2)
        return output

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AutoEncoder().to(device)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


train_loss = []
print(model)
old_loss = 100000
for i in tqdm(range(50)):
    epoch_loss = 0
    for j, signal in enumerate(x_test_data):
        x = signal.to(device)
        
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output, x)
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
    cost = epoch_loss / len(x_test_data)
    print(cost)
    train_loss.append(cost)
    if cost < old_loss:
        print("ok")
        old_loss = cost
        torch.save(model, 'AE.pt')