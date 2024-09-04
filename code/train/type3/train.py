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

level = 'document'


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

#xn = train_text
#test_xn = test_text

xn = np.concatenate([train_text, train_key], axis=1)
test_xn = np.concatenate([test_text, test_key], axis=1)



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
    log_text = open(f"type3/logfiles/{level}.txt",'a')
    precision= 0
    recall = 0
    f1 = 0
    log_text.write(str(t)+'  ')
    #5188*300, 1297*300, 5188*1, 1297*1
    #data_y:6485*96
    X_train, X_val, y_train, y_val = train_test_split(xn, data_y[t], test_size=0.2)
    X_test = test_xn
    y_test = test_data_y[t]

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
    #joblib.dump(clf, 'checkpoints/document/'+str(t))
    y_hat = clf.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_hat, average='binary')
    log_text.write(str(precision)+' ')
    log_text.write(str(recall)+' ')
    log_text.write(str(f1)+' \n')
    f1s.append(f1)
    log_text.close()


log_text = open(f"type3/logfiles/{level}.txt",'a')
log_text.write('average f1 for level1: '+str(np.mean(f1s[:14]))+'\n')
log_text.write('average f1: '+str(np.mean(f1s))+'\n')
log_text.close()

