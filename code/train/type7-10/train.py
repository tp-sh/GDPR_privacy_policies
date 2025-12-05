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
parser.add_argument('--aim', type=str, default='2_6') #1, 2_2, 2_3, 2_4, 2_5, 2_6
parser.add_argument('--N_EPOCHS', type=int, default=10)
parser.add_argument('--ctype', type=str, default='1')
parser.add_argument('--on_gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--fold', type=str, default='1')
args = parser.parse_args()

level = args.level
aim = args.aim
N_EPOCHS = args.N_EPOCHS
on_gpu = [args.on_gpu]
ctype = args.ctype
lr = args.lr
fold = args.fold

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

data["text"] = data["text"].str.lower()
data["text"] = data["text"].apply(cleanPunc)
data["text"] = data["text"].apply(keepAlpha)

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

data["text"] = data["text"].apply(text_clean)

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

print("Generating GloVe embeddings for training data...")
data_X = np.array([get_glove_embedding(text, glove_embeddings, EMBEDDING_DIM) for text in data['text']])
data_y = data.drop(labels=["label", "text","parents","siblings","parents_matrix","siblings_matrix"], axis=1)
data_y = data_y[LABEL_COLUMNS]

parents_text = train_df['parents'].copy()
parents_text = parents_text.astype(str).str.lower()
parents_text = parents_text.apply(cleanPunc)
parents_text = parents_text.apply(keepAlpha)
parents_text = parents_text.apply(text_clean)
df2n = np.array([get_glove_embedding(text, glove_embeddings, EMBEDDING_DIM) for text in parents_text])

siblings_text = train_df['siblings'].copy()
siblings_text = siblings_text.astype(str).str.lower()
siblings_text = siblings_text.apply(cleanPunc)
siblings_text = siblings_text.apply(keepAlpha)
siblings_text = siblings_text.apply(text_clean)
df3n = np.array([get_glove_embedding(text, glove_embeddings, EMBEDDING_DIM) for text in siblings_text])

key_path = 'keyword.txt'
keywordfile = open(key_path)
keywords = keywordfile.readlines()
key_list = {}
for t in range(0,95):
    keywordss = keywords[t].replace('\n','')
    key_list[t] = keywordss.split(';')

train_key = []
for i in range(len(train_df)):
    list = ['0'] * 95
    strr = train_df.iloc[i]['text']
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
    train_key.append(key_str)

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

list1_train = []
for item in train_key:
    itemlist = []
    item_str = str(item) if not pd.isna(item) else ""
    itemstr = re.findall(r'\d+', item_str)
    for t in itemstr:
        itemlist.append(int(t))
    itemnp = np.array(itemlist)
    list1_train.append(itemnp)
dfkeyn_train = np.array(list1_train)

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


if ctype == '1':
    x_train = data_X
elif ctype == '2':
    x_train = np.concatenate([data_X, df2n, df3n], axis=1)
elif ctype == '3':
    x_train = np.concatenate([data_X, dfkeyn_train], axis=1)
elif ctype == '4':
    x_train = np.concatenate([data_X, df2n, df3n, dfkeyn_train], axis=1)

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
test_data_y = test_y[LABEL_COLUMNS]

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

model = NeuralNet(x_train.shape[1], len(LABEL_COLUMNS))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

class Mydataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = Mydataset(x_train, data_y.values)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

test_dataset = Mydataset(x_test, test_data_y.values)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

with open(f'Glove_nn/logfiles/{level}/{aim}.txt','a') as log:
    log.write(f'\nglove+nn_mlp, {level}, fold{fold}, type{ctype}, lr{lr}\n')

for epoch in range(N_EPOCHS):
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
    checkpoint_path = f'Glove_nn/checkpoints/glove_nn_notune_mlp_{fold}_{ctype}_{lr}/{level}/{aim}'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    torch.save(model.state_dict(), f'{checkpoint_path}/epoch_{epoch}.pt')
    old_pt = f"{checkpoint_path}/epoch_{epoch - 2}.pt"
    if os.path.exists(old_pt):
        os.remove(old_pt)
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

        f1s = []
        for i in range(len(LABEL_COLUMNS)):
            labelss = torch.cat(all_labels, dim=0)[:, i]
            predictionss = torch.cat(all_outputs, dim=0)[:, i]
            f1 = f1_score(labelss, predictionss, average='binary')
            f1s.append(f1)

    with open(f'Glove_nn/logfiles/{level}/{aim}.txt','a') as log:
        log.write(f"epoch {epoch} {np.mean(f1s)} {f1s}\n")


        

