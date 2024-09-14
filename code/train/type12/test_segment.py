import pandas as pd
import numpy as np
import io
import sklearn
import torch
import torch.nn as nn
import os
import csv
import nltk
import torchmetrics
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import joblib
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score
from train import PrivacyTagger
from transformers import AutoTokenizer
from matplotlib import re
from torch.utils.data import Dataset, DataLoader
from torch import optim, nn, utils, Tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

LABEL_COLUMNS1 =  ['1', '22', '38', '39', '41', '47', '54', '64', '65', '67', '85', '86', '90', '93']
LABEL_COLUMNS2 =  ['2', '4']
LABEL_COLUMNS3 =  ['23', '27', '28', '29', '30', '31', '32', '34']
LABEL_COLUMNS4 =  ['48', '49']
LABEL_COLUMNS5 =  ['55', '60', '62', '63']
LABEL_COLUMNS6 =  ['87', '88', '89']
LABEL_COLUMNS7 =  ['91', '92']

LABEL_COLUMNS = LABEL_COLUMNS1 + LABEL_COLUMNS2 + LABEL_COLUMNS3 + LABEL_COLUMNS4 + LABEL_COLUMNS5 + LABEL_COLUMNS6 + LABEL_COLUMNS7

modelP_level1 = PrivacyTagger.load_from_checkpoint('type6/checkpoints/segment/1/epoch_9.ckpt', n_classes=14, logfile_path='').to(device)
#modelP = AutoModel.from_pretrained("mukund/privbert")
modelT_level1 = modelP_level1.to(device)
modelT_level1.eval()

modelP_level2_1 = PrivacyTagger.load_from_checkpoint('type6/checkpoints/segment/2_1/epoch_9.ckpt', n_classes=2, logfile_path='').to(device)
#modelP = AutoModel.from_pretrained("mukund/privbert")
modelT_level2_1 = modelP_level2_1.to(device)
modelT_level2_1.eval()

modelP_level2_2 = PrivacyTagger.load_from_checkpoint('type6/checkpoints/segment/2_2/epoch_9.ckpt', n_classes=8, logfile_path='').to(device)
#modelP = AutoModel.from_pretrained("mukund/privbert")
modelT_level2_2 = modelP_level2_2.to(device)
modelT_level2_2.eval()


modelP_level2_3 = PrivacyTagger.load_from_checkpoint('type6/checkpoints/segment/2_3/epoch_7.ckpt', n_classes=2, logfile_path='').to(device)
#modelP = AutoModel.from_pretrained("mukund/privbert")
modelT_level2_3 = modelP_level2_3.to(device)
modelT_level2_3.eval()


modelP_level2_4 = PrivacyTagger.load_from_checkpoint('type6/checkpoints/segment/2_4/epoch_9.ckpt', n_classes=4, logfile_path='').to(device)
#modelP = AutoModel.from_pretrained("mukund/privbert")
modelT_level2_4 = modelP_level2_4.to(device)
modelT_level2_4.eval()


modelP_level2_5 = PrivacyTagger.load_from_checkpoint('type6/checkpoints/segment/2_5/epoch_7.ckpt', n_classes=3, logfile_path='').to(device)
#modelP = AutoModel.from_pretrained("mukund/privbert")
modelT_level2_5 = modelP_level2_5.to(device)
modelT_level2_5.eval()


modelP_level2_6 = PrivacyTagger.load_from_checkpoint('type6/checkpoints/segment/2_6/epoch_9.ckpt', n_classes=2, logfile_path='').to(device)
#modelP = AutoModel.from_pretrained("mukund/privbert")
modelT_level2_6 = modelP_level2_6.to(device)
modelT_level2_6.eval()


tokenizer = AutoTokenizer.from_pretrained("/data/data1/cyx/privbert")
data = pd.read_csv('/data/data1/cyx/p_dataset_segment_test.csv')

def cleanPunc(sentence):
    """删除符号"""
    sentence = str(sentence)
    cleaned = re.sub(r'[?|!|\'|"|#]', r" ", sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r" ", cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned

def keepAlpha(sentence):
    """删除字母和空格以外的所有词"""
    sentence = str(sentence)
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub("[^a-z A-Z]+", " ", word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


def text_clean(text):
    # 用空格替换各种符号

    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    NonSTOPWORDS = []
    # 保留一些有意义的停止词
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join([ w for w in text.split() if ((w not in STOPWORDS) or (w in NonSTOPWORDS))])
    return text

data["text"] = data["text"].str.lower()
data["text"] = data["text"].apply(cleanPunc)
data["text"] = data["text"].apply(keepAlpha)
data["text"] = data["text"].apply(text_clean)

data["parents"] = data["parents"].str.lower()
data["parents"] = data["parents"].apply(cleanPunc)
data["parents"] = data["parents"].apply(keepAlpha)
data["parents"] = data["parents"].apply(text_clean)

data["siblings"] = data["siblings"].str.lower()
data["siblings"] = data["siblings"].apply(cleanPunc)
data["siblings"] = data["siblings"].apply(keepAlpha)
data["siblings"] = data["siblings"].apply(text_clean)

class PrivacyDataset(Dataset):              
    def __init__(self, data:pd.DataFrame, tokenizer, max_token_len = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        
        text = data_row.text
        parents = data_row.parents
        siblings = data_row.siblings
        labels = data_row[LABEL_COLUMNS]
        
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        encoding2 = self.tokenizer.encode_plus(
            parents,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        encoding3 = self.tokenizer.encode_plus(
            siblings,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        ) 

        return dict(
            text=text,
            parents = parents,
            siblings = siblings,
            input_ids=encoding['input_ids'].flatten(),
            attention_mask=encoding['attention_mask'].flatten(),
            input_ids2=encoding2['input_ids'].flatten(),
            attention_mask2=encoding2['attention_mask'].flatten(),
            input_ids3=encoding3['input_ids'].flatten(),
            attention_mask3=encoding3['attention_mask'].flatten(),
            labels=torch.FloatTensor(labels)
        )

test_dataset = PrivacyDataset(data, tokenizer, max_token_len=256)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

all_models_outputs = []
all_models_labels = []

for i in range(7):
    model = [modelT_level1, modelT_level2_1, modelT_level2_2, modelT_level2_3, modelT_level2_4, modelT_level2_5, modelT_level2_6][i]
    with torch.no_grad():
        all_outputs = []
        all_labels = []
        for j, data in enumerate(test_loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)

            input_ids2 = data['input_ids2'].to(device)
            attention_mask2 = data['attention_mask2'].to(device)

            input_ids3 = data['input_ids3'].to(device)
            attention_mask3 = data['attention_mask3'].to(device)

            labels = data['labels'].detach().cpu()
            outputs = model(input_ids, attention_mask, input_ids2, attention_mask2, input_ids3, attention_mask3)[1].detach().cpu()
            
            if i == 0:
                labels = labels[:, :14]
            elif i == 1:
                labels = labels[:, 14:16]
            elif i == 2:
                labels = labels[:, 16:24]
            elif i == 3:
                labels = labels[:, 24:26]
            elif i == 4:
                labels = labels[:, 26:30]
            elif i == 5:
                labels = labels[:, 30:33]
            elif i == 6:
                labels = labels[:, 33:35]

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
    #根据all_models_outputs[0]的结果修正all_models_outputs[i]的结果
    #如果all_models_outputs[0][j][0]==0,那么对应的all_models_outputs[i][j]的每一位都是0

    for j in range(len(all_outputs)):
        if all_models_outputs[0][j][0][pre]==0:
            all_outputs[j] = torch.zeros_like(all_outputs[j])
    for j in range(len([LABEL_COLUMNS1, LABEL_COLUMNS2, LABEL_COLUMNS3, LABEL_COLUMNS4, LABEL_COLUMNS5, LABEL_COLUMNS6, LABEL_COLUMNS7][i])):
        labelss = torch.cat(all_labels, dim=0)[:, j]
        predictionss = torch.cat(all_outputs, dim=0)[:, j]
        f1 = f1_score(labelss, predictionss, average='binary')
        f1s.append(f1)
    


with open(f'type6/logfiles/segment/results.txt','a') as log:
    log.write('f1s: '+str(f1s)+'\n')
    log.write('average f1 for level1: '+str(np.mean(f1s[:14]))+'\n')
    log.write('average f1: '+str(np.mean(f1s))+'\n')
    
