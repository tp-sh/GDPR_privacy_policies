# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sklearn
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import nltk
import torchmetrics
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys
import warnings
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from matplotlib import re
import pytorch_lightning as pl
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import optim, nn, utils, Tensor
import torch.multiprocessing
import argparse
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser()
parser.add_argument('--level', type=str, default='segment') #segment
parser.add_argument('--aim', type=str, default='2_1') #1, 2_2, 2_3, 2_4, 2_5, 2_6
parser.add_argument('--N_EPOCHS', type=int, default=10)
parser.add_argument('--on_gpu', type=int, default=0)
args = parser.parse_args()

level = args.level
aim = args.aim
N_EPOCHS = args.N_EPOCHS
on_gpu = [args.on_gpu]

mark = 'type5: '+aim+' '+level+'level'


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


if level == 'document':
    file_path = "/home/user/cthlx/new_version/p_dataset_matrix_119.csv"
    df = pd.read_csv(file_path)
elif level == 'segment':
    file_path = '/data/data1/cyx/p_dataset_segment_train.csv'
    df = pd.read_csv(file_path)
    
df.fillna(0,inplace=True)
df.drop(index=(df.loc[(df['text']==0)].index))
for x in df.index:
    data_flag = False
    for t in LABEL_COLUMNS:
        if df.loc[x,t] != 0:
            data_flag = True
    if data_flag == False:
        df.drop(x, inplace = True)
print(df.shape)
data = df.loc[np.random.choice(df.index, size=df.shape[0])]
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
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

data["text"] = data["text"].str.lower()
data["text"] = data["text"].apply(cleanPunc)
data["text"] = data["text"].apply(keepAlpha)


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

data["text"] = data["text"].apply(text_clean)



from sklearn.model_selection import train_test_split #train-test split
train_df, val_df = train_test_split(data,test_size = 0.2)


class PrivacyDataset(Dataset):              
    def __init__(self, data:pd.DataFrame, tokenizer, max_token_len = 256):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained("/data/data1/cyx/privbert")
        self.max_token_len = max_token_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        text = data_row.text
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

        return dict(
            text=text,
            input_ids=encoding['input_ids'].flatten(),
            attention_mask=encoding['attention_mask'].flatten(),
            labels=torch.FloatTensor(labels)
        )
    
class PrivacyDataModule(pl.LightningDataModule):
    
    def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=256):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        
    def setup(self,stage):
        self.train_dataset = PrivacyDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )
        
        self.test_dataset = PrivacyDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=4
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=4
        )
    
BATCH_SIZE = 32
device = torch.device('cuda')

class PrivacyTagger(pl.LightningModule):
    def __init__(self, n_classes: int, steps_per_epoch=None, n_epochs=None):
        super().__init__()
        self.validation_step_outputs = []
        self.bert = AutoModel.from_pretrained("/data/data1/cyx/privbert")
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, n_classes)
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.criterion = nn.BCELoss()
        
    def forward(self, input_ids, attention_mask, labels=None):
        output1 = self.bert(input_ids, attention_mask=attention_mask)
        #print(output4.shape)
        output = self.classifier(output1.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    def training_step(self,batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self(input_ids, attention_mask, labels)

        return {'loss': loss, 'predictions': outputs, 'labels': labels}
    
    def validation_step(self,batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self(input_ids, attention_mask, labels)
        self.validation_step_outputs.append({"predictions": outputs, "labels": labels})
        return {'loss': loss, 'predictions': outputs, 'labels': labels}
    
    def test_step(self,batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        labels = batch['labels']
        loss, outputs = self(input_ids, attention_mask, labels)
        return loss
    
    def validation_epoch_end(self, outputs):
        
        labels = []
        predictions = []
        
        for out in self.validation_step_outputs:
            #print(out)
           
            for out_labels in out['labels'].detach().cpu():
                labels.append(out_labels)
                
            for out_predictions in out['predictions'].detach().cpu():
                predictions.append(out_predictions)
   
        labels = torch.stack(labels)
        predictions = torch.stack(predictions)

        f1s = []
        for i, name in enumerate(LABEL_COLUMNS):
            labelss = labels[:, i].int()
            numm = torch.tensor(0.5)
            predictionss = (predictions[:,i] + numm).int()

            f1 = f1_score(labelss, predictionss, average='binary')
            f1s.append(f1)

        with open(f'type5/logfiles/{level}/{aim}.txt','a') as logfile:
            logfile.write(mark+'\n')
            logfile.write('f1s: '+str(np.mean(f1s)) + str(f1s)+'\n')

        self.validation_step_outputs.clear()   
       
        
            
    def configure_optimizers(self):
        
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
 

from pytorch_lightning.callbacks import Callback

class SaveCheckpointCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        trainer.save_checkpoint(f"type5/checkpoints/{level}/{aim}/epoch_{trainer.current_epoch}.ckpt")

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("/data/data1/cyx/privbert")
    data_module = PrivacyDataModule(train_df,val_df,tokenizer)
    data_module.setup(stage=None)
    model = PrivacyTagger(
        n_classes=len(LABEL_COLUMNS),
        steps_per_epoch=len(train_df) // BATCH_SIZE,
        n_epochs=N_EPOCHS
        )

    CUDA_VISIBLE_DEVICES = 0
    trainer = pl.Trainer(max_epochs=N_EPOCHS, accelerator="gpu", gpus=on_gpu, callbacks=[SaveCheckpointCallback()])

    trainer.fit(model, data_module)