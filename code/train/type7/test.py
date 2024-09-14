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

parser = argparse.ArgumentParser()
parser.add_argument('--level', type=str, default='segment') #segment
parser.add_argument('--on_gpu', type=int, default=0)
args = parser.parse_args()
level = args.level
on_gpu = [args.on_gpu]


pt1 = torch.load(f'type7/checkpoints/{level}/1/epoch_9.pt')
pt2 = torch.load(f'type7/checkpoints/{level}/2_1/epoch_9.pt')
pt3 = torch.load(f'type7/checkpoints/{level}/2_2/epoch_9.pt')
pt4 = torch.load(f'type7/checkpoints/{level}/2_3/epoch_9.pt')
pt5 = torch.load(f'type7/checkpoints/{level}/2_4/epoch_9.pt')
pt6 = torch.load(f'type7/checkpoints/{level}/2_5/epoch_9.pt')
pt7 = torch.load(f'type7/checkpoints/{level}/2_6/epoch_9.pt')

test_data_y = joblib.load(f'Embedding/TFIDF_Embeddings/{level}/test_y.pkl')
test_text = joblib.load(f'Embedding/TFIDF_Embeddings/{level}/test_text.pkl')
test_parents = joblib.load(f'Embedding/TFIDF_Embeddings/{level}/test_parents.pkl')
test_siblings = joblib.load(f'Embedding/TFIDF_Embeddings/{level}/test_siblings.pkl')
test_key = joblib.load(f'Embedding/TFIDF_Embeddings/{level}/test_key.pkl')

test_xn = test_text


LABEL_COLUMNS1 = ['1', '22', '38', '39', '41', '47', '54', '64', '65', '67', '85', '86', '90', '93']
LABEL_COLUMNS2 = ['2', '4']
LABEL_COLUMNS3 = ['23', '27', '28', '29', '30', '31', '32', '34']
LABEL_COLUMNS4 = ['48', '49']
LABEL_COLUMNS5 = ['55', '60', '62', '63']
LABEL_COLUMNS6 = ['87', '88', '89']
LABEL_COLUMNS7 = ['91', '92']



test_data_y1 = test_data_y[LABEL_COLUMNS1]
test_data_y2 = test_data_y[LABEL_COLUMNS2]
test_data_y3 = test_data_y[LABEL_COLUMNS3]
test_data_y4 = test_data_y[LABEL_COLUMNS4]
test_data_y5 = test_data_y[LABEL_COLUMNS5]
test_data_y6 = test_data_y[LABEL_COLUMNS6]
test_data_y7 = test_data_y[LABEL_COLUMNS7]

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
    
model1 = NeuralNet(test_xn.shape[1], len(LABEL_COLUMNS1))
model1.load_state_dict(pt1)

model2_1 = NeuralNet(test_xn.shape[1], len(LABEL_COLUMNS2))
model2_1.load_state_dict(pt2)

model2_2 = NeuralNet(test_xn.shape[1], len(LABEL_COLUMNS3))
model2_2.load_state_dict(pt3)

model2_3 = NeuralNet(test_xn.shape[1], len(LABEL_COLUMNS4))
model2_3.load_state_dict(pt4)

model2_4 = NeuralNet(test_xn.shape[1], len(LABEL_COLUMNS5))
model2_4.load_state_dict(pt5)

model2_5 = NeuralNet(test_xn.shape[1], len(LABEL_COLUMNS6))
model2_5.load_state_dict(pt6)

model2_6 = NeuralNet(test_xn.shape[1], len(LABEL_COLUMNS7))
model2_6.load_state_dict(pt7)


model1.eval()
model2_1.eval()
model2_2.eval()
model2_3.eval()
model2_4.eval()
model2_5.eval()
model2_6.eval()


class Mydataset(torch.utils.data.Dataset):
    def __init__(self, X, y1, y2, y3, y4, y5, y6, y7):
        self.X = X
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4
        self.y5 = y5
        self.y6 = y6
        self.y7 = y7

    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y1[idx], self.y2[idx], self.y3[idx], self.y4[idx], self.y5[idx], self.y6[idx], self.y7[idx]

test_dataset = Mydataset(test_xn, test_data_y1.values, test_data_y2.values, test_data_y3.values, test_data_y4.values, test_data_y5.values, test_data_y6.values, test_data_y7.values)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

all_models_outputs = []
all_models_labels = []
for i in range(7):
    model = [model1, model2_1, model2_2, model2_3, model2_4, model2_5, model2_6][i]
    with torch.no_grad():
        all_outputs = []
        all_labels = []
        for j, (inputs, labels1, labels2, labels3, labels4, labels5, labels6, labels7) in enumerate(test_loader):
            inputs = inputs.float()
            labels1 = labels1.float()
            labels2 = labels2.float()
            labels3 = labels3.float()
            labels4 = labels4.float()
            labels5 = labels5.float()
            labels6 = labels6.float()
            labels7 = labels7.float()
            
            labels = [labels1, labels2, labels3, labels4, labels5, labels6, labels7][i]
            outputs = model(inputs)
            outputs = (outputs>=0.5).int()
            labels = labels.int()
            all_outputs.append(outputs)
            all_labels.append(labels)
    all_models_outputs.append(all_outputs)
    all_models_labels.append(all_labels)

f1s = []
all_outputs = all_models_outputs[0]
all_labels = all_models_labels[0]

for i in range(len(LABEL_COLUMNS1)):
    labelss = torch.cat(all_labels, dim=0)[:, i]
    predictionss = torch.cat(all_outputs, dim=0)[:, i]
    f1 = f1_score(labelss, predictionss, average='binary')
    f1s.append(f1)


pres = [0, 1, 5, 6, 11, 12]
for i in range(1,7):
    all_outputs = all_models_outputs[i]
    all_labels = all_models_labels[i]
    pre = pres[i-1]

    for j in range(len(all_outputs)):
        if all_models_outputs[0][j][0][pre]==0:
            all_outputs[j] = torch.zeros_like(all_outputs[j])
    for j in range(len([LABEL_COLUMNS1, LABEL_COLUMNS2, LABEL_COLUMNS3, LABEL_COLUMNS4, LABEL_COLUMNS5, LABEL_COLUMNS6, LABEL_COLUMNS7][i])):
        labelss = torch.cat(all_labels, dim=0)[:, j]
        predictionss = torch.cat(all_outputs, dim=0)[:, j]
        f1 = f1_score(labelss, predictionss, average='binary')
        f1s.append(f1)
    


with open(f'type7/logfiles/{level}/results.txt','a') as log:
    log.write('f1s: '+str(f1s)+'\n')
    log.write('average f1 for level1: '+str(np.mean(f1s[:14]))+'\n')
    log.write('average f1: '+str(np.mean(f1s))+'\n')
    
