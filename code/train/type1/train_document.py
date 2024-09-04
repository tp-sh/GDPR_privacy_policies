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

#document level

file_path = "/data/data1/cyx/p_dataset_matrix_119.csv"
df = pd.read_csv(file_path)
print("# of rows in data = {}".format(df.shape[0]))
print("# of columns in data = {}".format(df.shape[1]))


df.fillna(0,inplace=True)
df.drop(index=(df.loc[(df['text']==0)].index))
missing_values_check = df.isnull().sum()

data = df.loc[np.random.choice(df.index, size=df.shape[0])]

test_file_path = "/data/data1/cyx/p_dataset_matrix_30.csv"
test_df = pd.read_csv(test_file_path)
test_df.fillna(0,inplace=True)
test_df.drop(index=(test_df.loc[(test_df['text']==0)].index))
test_missing_values_check = test_df.isnull().sum()

test_data = test_df.loc[np.random.choice(test_df.index, size=test_df.shape[0])]

import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re
import sys
import warnings


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



test_data["text"] = test_data["text"].str.lower()
test_data["text"] = test_data["text"].apply(cleanPunc)
test_data["text"] = test_data["text"].apply(keepAlpha)



def text_clean(text):
    # 用空格替换各种符号
    lem = WordNetLemmatizer()  # 词性还原
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    #NonSTOPWORDS = ['what', 'why', 'how', 'when', 'who', 'with', 'about', 'from', 'we', 'our', 'you', 'your']
    NonSTOPWORDS = []
    # 保留一些有意义的停止词
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join([ w for w in text.split() if ((w not in STOPWORDS) or (w in NonSTOPWORDS))])
    return text

data["text"] = data["text"].apply(text_clean)


test_data["text"] = test_data["text"].apply(text_clean)


#PrivBert
#tokenizer = AutoTokenizer.from_pretrained("/data/data1/cyx/privbert")
#bert = AutoModel.from_pretrained("/data/data1/cyx/privbert")

#TFIDF

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", ngram_range=(1,3), norm="l2")
vectorizer.fit(data['text'])
#joblib.dump(vectorizer, 'checkpoints/document/p_vector_vector1.pkl')
#6485*170010(经过tdidf向量化后)
data_X = vectorizer.transform(data['text'])
test_data_X = vectorizer.transform(test_data['text'])
data_y = data.drop(labels=["label", "text","parents","siblings","parents_matrix","siblings_matrix"], axis=1)
test_data_y = test_data.drop(labels=["label", "text","parents","siblings","parents_matrix","siblings_matrix"], axis=1)
#6485*96

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
svd = TruncatedSVD(300)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
lsa.fit(data_X)
#joblib.dump(lsa, 'checkpoints/document/p_lsa_vector1')
data_X = lsa.transform(data_X)
test_data_X = lsa.transform(test_data_X)
#6485*300




xn = data_X
test_xn = test_data_X



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
    #log_text = open("logfiles/document.txt",'a')
    precision= 0
    recall = 0
    f1 = 0
    #log_text.write(str(t)+'  ')
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
        #log_text.write('0 ')
        #log_text.write('0 ')
        #log_text.write('0 \n ')
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
    joblib.dump(clf, 'checkpoints/document/'+str(t))
    y_hat = clf.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_hat, average='binary')
    #log_text.write(str(precision)+' ')
    #log_text.write(str(recall)+' ')
    #log_text.write(str(f1)+' \n')
    f1s.append(f1)
    #log_text.close()


log_text = open("logfiles/document.txt",'a')
log_text.write('average f1 for level1: '+str(np.mean(f1s[:14]))+'\n')
log_text.write('average f1: '+str(np.mean(f1s))+'\n')
log_text.close()

