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

#segment level

train_text_path = '/data/data1/cyx/Embedding/PrivBert_Embeddings3/segment/train_text.pt'
test_text_path = '/data/data1/cyx/Embedding/PrivBert_Embeddings3/segment/test_text.pt'

train_parents_path = '/data/data1/cyx/Embedding/PrivBert_Embeddings3/segment/train_parents.pt'
test_parents_path = '/data/data1/cyx/Embedding/PrivBert_Embeddings3/segment/test_parents.pt'

train_siblings_path = '/data/data1/cyx/Embedding/PrivBert_Embeddings3/segment/train_siblings.pt'
test_siblings_path = '/data/data1/cyx/Embedding/PrivBert_Embeddings3/segment/test_siblings.pt'

train_text_x, train_text_y = torch.load(train_text_path)
test_text_x, test_text_y = torch.load(test_text_path)

train_parents_x, train_parents_y = torch.load(train_parents_path)
test_parents_x, test_parents_y = torch.load(test_parents_path)

train_siblings_x, train_siblings_y = torch.load(train_siblings_path)
test_siblings_x, test_siblings_y = torch.load(test_siblings_path)


x = np.concatenate([train_text_x, train_parents_x], axis=1)
#(6484,2304)
xn = np.concatenate([x, train_siblings_x], axis=1)

test_x = np.concatenate([test_text_x, test_parents_x], axis=1)
#(1480,2304)
test_xn = np.concatenate([test_x, test_siblings_x], axis=1)

data_y = train_text_y
test_data_y = test_text_y



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.over_sampling import SMOTE 
from collections import Counter


LABEL_COLUMNS = []
modellist = ['1', '22', '38', '39', '41', '47', '54', '64', '65', '67', '85', '86', '90', '93',  '2', '4', '23', '27', '28', '29', '30', '31', '32', '34','48', '49','55', '60', '62', '63','87', '88', '89','91', '92']
for i in range(1,97):
    LABEL_COLUMNS.append(str(i))

f1s = []
for t in modellist:
    log_text = open("type12/logfiles/segment.txt",'a')
    precision= 0
    recall = 0
    f1 = 0
    log_text.write(str(t)+'  ')
    
    X_train, X_val, y_train, y_val = train_test_split(xn, data_y[:,int(t)-1], test_size=0.2)
    X_test = test_xn
    y_test = test_data_y[:,int(t)-1]

    dicty = Counter(y_train)
    if dicty[1] >= 1000:
        X_resampled2 = X_train
        y_resampled2 = y_train
    elif dicty[1] >= 500:
        samplerate = float(0.5)
        Rus = RandomUnderSampler(sampling_strategy=samplerate)
        X_resampled2, y_resampled2 = Rus.fit_resample(X_train,y_train)
    elif dicty[1] == 0:
        log_text.write('0 ')
        log_text.write('0 ')
        log_text.write('0 \n ')
        continue
    elif dicty[1] <= 50:
        samplerate = float(0.5)
        Rus = RandomUnderSampler(sampling_strategy=samplerate)
        X_resampled2, y_resampled2 = Rus.fit_resample(X_train,y_train)
    else:
        sampledict = {0:1000}
        Rus = RandomUnderSampler(sampling_strategy=sampledict)
        X_resampled, y_resampled = Rus.fit_resample(X_train,y_train)
        sampledict2 = {1:500}
        smt = SMOTE(sampling_strategy=sampledict2)
        X_resampled2, y_resampled2 = smt.fit_resample(X_resampled, y_resampled)

    #random forest
    clf = RandomForestClassifier(max_depth=None, random_state=50)
    clf.fit(X_resampled2, y_resampled2)
    #joblib.dump(clf, 'type12/checkpoints/segment/'+str(t))
    y_hat = clf.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_hat, average='binary')
    log_text.write(str(precision)+' ')
    log_text.write(str(recall)+' ')
    log_text.write(str(f1)+' \n')
    f1s.append(f1)
    log_text.close()


log_text = open("type12/logfiles/segment.txt",'a')
log_text.write('average f1 for level1: '+str(np.mean(f1s[:14]))+'\n')
log_text.write('average f1: '+str(np.mean(f1s))+'\n')
log_text.close()
