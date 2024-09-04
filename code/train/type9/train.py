import pandas as pd
import numpy as np
import io
import sklearn
import torch
import torch.nn as nn
import os
import csv
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import joblib
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score

lr = 0.05
parser = argparse.ArgumentParser()
parser.add_argument('--level', type=str, default='segment') #segment
parser.add_argument('--aim', type=str, default='2_6') #1, 2_2, 2_3, 2_4, 2_5, 2_6
parser.add_argument('--N_EPOCHS', type=int, default=10)
parser.add_argument('--on_gpu', type=int, default=0)
args = parser.parse_args()

level = args.level
aim = args.aim
N_EPOCHS = args.N_EPOCHS
on_gpu = [args.on_gpu]


if aim == '1':
    LABEL_COLUMNS = ['1', '22', '38', '39', '41', '47', '54', '64', '65', '67', '85', '86', '90', '93']
elif aim == '2_1':
    LABEL_COLUMNS = ['2', '4']
elif aim == '2_2':
    LABEL_COLUMNS = ['23', '27', '28', '29', '30', '31', '32', '34']
elif aim == '2_3':
    LABEL_COLUMNS = ['48', '49']
elif aim == '2_4':
    LABEL_COLUMNS = ['55', '60', '62', '63']
elif aim == '2_5':
    LABEL_COLUMNS = ['87', '88', '89']
elif aim == '2_6':
    LABEL_COLUMNS = ['91', '92']


data_y = joblib.load(f'/data/data1/cyx/Embedding/TFIDF_Embeddings/{level}/train_y.pkl')
test_data_y = joblib.load(f'/data/data1/cyx/Embedding/TFIDF_Embeddings/{level}/test_y.pkl')
train_text = joblib.load(f'/data/data1/cyx/Embedding/TFIDF_Embeddings/{level}/train_text.pkl')
test_text = joblib.load(f'/data/data1/cyx/Embedding/TFIDF_Embeddings/{level}/test_text.pkl')
train_parents = joblib.load(f'/data/data1/cyx/Embedding/TFIDF_Embeddings/{level}/train_parents.pkl')
test_parents = joblib.load(f'/data/data1/cyx/Embedding/TFIDF_Embeddings/{level}/test_parents.pkl')
train_siblings = joblib.load(f'/data/data1/cyx/Embedding/TFIDF_Embeddings/{level}/train_siblings.pkl')
test_siblings = joblib.load(f'/data/data1/cyx/Embedding/TFIDF_Embeddings/{level}/test_siblings.pkl')
train_key = joblib.load(f'/data/data1/cyx/Embedding/TFIDF_Embeddings/{level}/train_key.pkl')
test_key = joblib.load(f'/data/data1/cyx/Embedding/TFIDF_Embeddings/{level}/test_key.pkl')



xn = np.concatenate([train_text, train_key], axis=1)
test_xn = np.concatenate([test_text, test_key], axis=1)

data_y = data_y[LABEL_COLUMNS]
test_data_y = test_data_y[LABEL_COLUMNS]

#nn的方法

import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.classifier = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x
    
model = NeuralNet(xn.shape[1], len(LABEL_COLUMNS))

import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

#xn 6485*300，作为训练集的特征，data_y 6485*96，作为训练集的标签
#test_xn 1000*300，作为测试集的特征，test_data_y 1000*96，作为测试集的标签
#接下来使用batch训练的方法将数据喂给model并进行优化

class Mydataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = Mydataset(xn, data_y.values)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

test_dataset = Mydataset(test_xn, test_data_y.values)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


for epoch in range(N_EPOCHS):
    #train
    model.train()
    print(f"epoch {epoch}")
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.float()
        labels = labels.float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    if not os.path.exists(f'type9/checkpoints/{level}/{aim}'):
        os.mkdir(f'type9/checkpoints/{level}/{aim}')
    torch.save(model.state_dict(), f'type9/checkpoints/{level}/{aim}/epoch_{epoch}.pt')
        
    #test
    model.eval()
    with torch.no_grad():
        all_outputs = []
        all_labels = []
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.float()
            labels = labels.float()

            outputs = model(inputs)
            outputs = (outputs>=0.5).int()
            labels = labels.int()
            all_outputs.append(outputs)
            all_labels.append(labels)
        #outputs中的第一类的预测的precision,recall和f1

        f1s = []
        for i in range(len(LABEL_COLUMNS)):
            labelss = torch.cat(all_labels, dim=0)[:, i]
            predictionss = torch.cat(all_outputs, dim=0)[:, i]
            f1 = f1_score(labelss, predictionss, average='binary')
            f1s.append(f1)

    with open(f'type9/logfiles/{level}/{aim}.txt','a') as log:
        log.write(f"epoch {epoch} {np.mean(f1s)}\n")


        

