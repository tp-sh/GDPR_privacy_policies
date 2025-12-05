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
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--level', type=str, default='segment') #segment
parser.add_argument('--on_gpu', type=int, default=0)
parser.add_argument('--ctype', type=str, default='1')
parser.add_argument('--checkpoint', type=str, default='glove_nn_notune_mlp')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--fold', type=str, default='1')
args = parser.parse_args()
level = args.level
on_gpu = [args.on_gpu]
ctype = args.ctype
checkpoint = args.checkpoint
lr = args.lr
fold = args.fold

pt1 = torch.load(f'Glove_nn/checkpoints/{checkpoint}_{fold}_{ctype}_{lr}/{level}/1/epoch_9.pt')
pt2 = torch.load(f'Glove_nn/checkpoints/{checkpoint}_{fold}_{ctype}_{lr}/{level}/2_1/epoch_9.pt')
pt3 = torch.load(f'Glove_nn/checkpoints/{checkpoint}_{fold}_{ctype}_{lr}/{level}/2_2/epoch_9.pt')
pt4 = torch.load(f'Glove_nn/checkpoints/{checkpoint}_{fold}_{ctype}_{lr}/{level}/2_3/epoch_9.pt')
pt5 = torch.load(f'Glove_nn/checkpoints/{checkpoint}_{fold}_{ctype}_{lr}/{level}/2_4/epoch_9.pt')
pt6 = torch.load(f'Glove_nn/checkpoints/{checkpoint}_{fold}_{ctype}_{lr}/{level}/2_5/epoch_9.pt')
pt7 = torch.load(f'Glove_nn/checkpoints/{checkpoint}_{fold}_{ctype}_{lr}/{level}/2_6/epoch_9.pt')

train_file_path = f"/data/data1/cyx/cross_validation/{level}/fold_{fold}_train.csv"
train_df = pd.read_csv(train_file_path)
test_file_path = f"/data/data1/cyx/cross_validation/{level}/fold_{fold}_test.csv"
test_df = pd.read_csv(test_file_path)

train_df.fillna(0, inplace=True)
train_df = train_df.loc[~(train_df['text'] == 0)]
test_df.fillna(0, inplace=True)
test_df = test_df.loc[~(test_df['text'] == 0)]
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

data = train_df  

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def cleanPunc(sentence):
    sentence = str(sentence)
    cleaned = re.sub(r'[?|!|\'|"|#]', r" ", sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r" ", cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned

def keepAlpha(sentence):
    sentence = str(sentence)
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub("[^a-z A-Z]+", " ", word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

def text_clean(text):
    lem = WordNetLemmatizer()
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    NonSTOPWORDS = []
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join([ w for w in text.split() if ((w not in STOPWORDS) or (w in NonSTOPWORDS))])
    return text

GLOVE_PATH = "GloVefile/glove.6B.300d.txt" 

print("Loading GloVe vectors...")
glove_embeddings = {}
with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        glove_embeddings[word] = vector

EMBEDDING_DIM = len(next(iter(glove_embeddings.values())))
print(f"GloVe loaded. Embedding dim: {EMBEDDING_DIM}, vocab size: {len(glove_embeddings)}")

def get_glove_embedding(text, embeddings, dim):
    words = text.split()
    valid_vectors = []
    for word in words:
        if word in embeddings:
            valid_vectors.append(embeddings[word])
    if valid_vectors:
        return np.mean(valid_vectors, axis=0)
    else:
        return np.zeros(dim) 


key_path = 'keyword.txt'
keywordfile = open(key_path)
keywords = keywordfile.readlines()
key_list = {}
for t in range(0,95):
    keywordss = keywords[t].replace('\n','')
    key_list[t] = keywordss.split(';')

test_key = []
for i in range(len(test_df)):
    list = ['0'] * 95
    strr = test_df.iloc[i]['text']
    for t in range(0, 95):
        num = len(key_list[t])
        sum = 0
        if (type(strr) == str):
            for k in range(0, num):
                sum += strr.find(key_list[t][k])
            if sum + num != 0:
                list[t] = '1'
        else:
            continue
    key_str = ','.join(list)
    test_key.append(key_str)
keywordfile.close()

list1_test = []
for item in test_key:
    itemlist = []
    item_str = str(item) if not pd.isna(item) else ""
    itemstr = re.findall(r'\d+', item_str)
    for t in itemstr:
        itemlist.append(int(t))
    itemnp = np.array(itemlist)
    list1_test.append(itemnp)
dfkeyn_test = np.array(list1_test)

test_data = test_df.copy()
test_data["text"] = test_data["text"].str.lower()
test_data["text"] = test_data["text"].apply(cleanPunc)
test_data["text"] = test_data["text"].apply(keepAlpha)
test_data["text"] = test_data["text"].apply(text_clean)

parents_text_test = test_df['parents'].copy()
parents_text_test = parents_text_test.astype(str).str.lower()
parents_text_test = parents_text_test.apply(cleanPunc)
parents_text_test = parents_text_test.apply(keepAlpha)
parents_text_test = parents_text_test.apply(text_clean)
df2n_test = np.array([get_glove_embedding(text, glove_embeddings, EMBEDDING_DIM) for text in parents_text_test])

siblings_text_test = test_df['siblings'].copy()
siblings_text_test = siblings_text_test.astype(str).str.lower()
siblings_text_test = siblings_text_test.apply(cleanPunc)
siblings_text_test = siblings_text_test.apply(keepAlpha)
siblings_text_test = siblings_text_test.apply(text_clean)
df3n_test = np.array([get_glove_embedding(text, glove_embeddings, EMBEDDING_DIM) for text in siblings_text_test])

print("Generating GloVe embeddings for test data...")
test_X_lsa = np.array([get_glove_embedding(text, glove_embeddings, EMBEDDING_DIM) for text in test_data['text']])

if ctype == '1':
    x_test = test_X_lsa
elif ctype == '2':
    x_test = np.concatenate([test_X_lsa, df2n_test, df3n_test], axis=1)
elif ctype == '3':
    x_test = np.concatenate([test_X_lsa, dfkeyn_test], axis=1)
elif ctype == '4':
    x_test = np.concatenate([test_X_lsa, df2n_test, df3n_test, dfkeyn_test], axis=1)

test_y = test_df.drop(labels=["label", "text","parents","siblings","parents_matrix","siblings_matrix"], axis=1)


LABEL_COLUMNS1 = ['1', '22', '38', '39', '41', '47', '54', '64', '65', '67', '85', '86', '90', '93']
LABEL_COLUMNS2 = ['2', '4']
LABEL_COLUMNS3 = ['23', '27', '28', '29', '30', '31', '32', '34']
LABEL_COLUMNS4 = ['48', '49']
LABEL_COLUMNS5 = ['55', '60', '62', '63']
LABEL_COLUMNS6 = ['87', '88', '89']
LABEL_COLUMNS7 = ['91', '92']

test_data_y1 = test_y[LABEL_COLUMNS1]
test_data_y2 = test_y[LABEL_COLUMNS2]
test_data_y3 = test_y[LABEL_COLUMNS3]
test_data_y4 = test_y[LABEL_COLUMNS4]
test_data_y5 = test_y[LABEL_COLUMNS5]
test_data_y6 = test_y[LABEL_COLUMNS6]
test_data_y7 = test_y[LABEL_COLUMNS7]

import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        hidden_size = min(512, input_size)  
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
    
model1 = NeuralNet(x_test.shape[1], len(LABEL_COLUMNS1))
model1.load_state_dict(pt1)

model2_1 = NeuralNet(x_test.shape[1], len(LABEL_COLUMNS2))
model2_1.load_state_dict(pt2)

model2_2 = NeuralNet(x_test.shape[1], len(LABEL_COLUMNS3))
model2_2.load_state_dict(pt3)

model2_3 = NeuralNet(x_test.shape[1], len(LABEL_COLUMNS4))
model2_3.load_state_dict(pt4)

model2_4 = NeuralNet(x_test.shape[1], len(LABEL_COLUMNS5))
model2_4.load_state_dict(pt5)

model2_5 = NeuralNet(x_test.shape[1], len(LABEL_COLUMNS6))
model2_5.load_state_dict(pt6)

model2_6 = NeuralNet(x_test.shape[1], len(LABEL_COLUMNS7))
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

test_dataset = Mydataset(x_test, test_data_y1.values, test_data_y2.values, test_data_y3.values, test_data_y4.values, test_data_y5.values, test_data_y6.values, test_data_y7.values)
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
    

with open(f'Glove_nn/logfiles/{level}/results.txt','a') as log:
    log.write(f'\nglove+nn, {level}, fold{fold}, type{ctype}, model{checkpoint}, lr{lr}\n')
    log.write('f1s: '+str(f1s)+'\n')
    log.write('average f1 for level1: '+str(np.mean(f1s[:14]))+'\n')
    log.write('average f1: '+str(np.mean(f1s))+'\n')
    
