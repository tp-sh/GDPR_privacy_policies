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



level = 'segment'

if level == 'document':
    file_path = "/data/data1/cyx/p_dataset_matrix_119.csv"
    test_file_path = "/data/data1/cyx/p_dataset_matrix_30.csv"

elif level == 'segment':
    file_path = "/data/data1/cyx/p_dataset_segment_train.csv"
    test_file_path = "/data/data1/cyx/p_dataset_segment_test.csv"

df = pd.read_csv(file_path)
data = df
print("# of rows in data = {}".format(data.shape[0]))
print("# of columns in data = {}".format(data.shape[1]))


#df.fillna(0,inplace=True)
#df.drop(index=(df.loc[(df['text']==0)].index))
#missing_values_check = df.isnull().sum()

#data = df.loc[np.random.choice(df.index, size=df.shape[0])]

test_df = pd.read_csv(test_file_path)
test_data = test_df
#test_df.fillna(0,inplace=True)
#test_df.drop(index=(test_df.loc[(test_df['text']==0)].index))
#test_missing_values_check = test_df.isnull().sum()

#test_data = test_df.loc[np.random.choice(test_df.index, size=test_df.shape[0])]

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

data["parents"] = data["parents"].str.lower()
data["parents"] = data["parents"].apply(cleanPunc)
data["parents"] = data["parents"].apply(keepAlpha)

data["siblings"] = data["siblings"].str.lower()
data["siblings"] = data["siblings"].apply(cleanPunc)
data["siblings"] = data["siblings"].apply(keepAlpha)

test_data["text"] = test_data["text"].str.lower()
test_data["text"] = test_data["text"].apply(cleanPunc)
test_data["text"] = test_data["text"].apply(keepAlpha)

test_data["parents"] = test_data["parents"].str.lower()
test_data["parents"] = test_data["parents"].apply(cleanPunc)
test_data["parents"] = test_data["parents"].apply(keepAlpha)

test_data["siblings"] = test_data["siblings"].str.lower()
test_data["siblings"] = test_data["siblings"].apply(cleanPunc)
test_data["siblings"] = test_data["siblings"].apply(keepAlpha)

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
data["parents"] = data["parents"].apply(text_clean)
data["siblings"] = data["siblings"].apply(text_clean)

test_data["text"] = test_data["text"].apply(text_clean)
test_data["parents"] = test_data["parents"].apply(text_clean)
test_data["siblings"] = test_data["siblings"].apply(text_clean)


#TFIDF

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", ngram_range=(1,3), norm="l2")
vectorizer.fit(data['text'])
joblib.dump(vectorizer, f'tfidf_vectorizer/{level}/p_vector_vector1.pkl')
#6485*170010(经过tdidf向量化后)
data_X = vectorizer.transform(data['text'])
test_data_X = vectorizer.transform(test_data['text'])
data_y = data.drop(labels=["label", "text","parents","siblings","parents_matrix","siblings_matrix"], axis=1)
test_data_y = test_data.drop(labels=["label", "text","parents","siblings","parents_matrix","siblings_matrix"], axis=1)
#6485*96
joblib.dump(data_y, f'TFIDF_Embeddings/{level}/train_y.pkl')
joblib.dump(test_data_y, f'TFIDF_Embeddings/{level}/test_y.pkl')

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
svd = TruncatedSVD(300)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
lsa.fit(data_X)
joblib.dump(lsa, f'tfidf_vectorizer/{level}/p_lsa_vector1')
data_X = lsa.transform(data_X)
test_data_X = lsa.transform(test_data_X)
joblib.dump(data_X, f'TFIDF_Embeddings/{level}/train_text.pkl')
joblib.dump(test_data_X, f'TFIDF_Embeddings/{level}/test_text.pkl')
#6485*300

vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", ngram_range=(1,3), norm="l2")

vectorizer.fit(data['parents'])
joblib.dump(vectorizer, f'tfidf_vectorizer/{level}/p_vector_vector2.pkl')
df2n = vectorizer.transform(data['parents'])
test_df2n = vectorizer.transform(test_data['parents'])

svd = TruncatedSVD(100)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
lsa.fit(df2n)
joblib.dump(lsa, f'tfidf_vectorizer/{level}/p_lsa_vector2')
df2n = lsa.transform(df2n)
test_df2n = lsa.transform(test_df2n)
joblib.dump(df2n, f'TFIDF_Embeddings/{level}/train_parents.pkl')
joblib.dump(test_df2n, f'TFIDF_Embeddings/{level}/test_parents.pkl')

vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", ngram_range=(1,3), norm="l2")
vectorizer.fit(data['siblings'])
joblib.dump(vectorizer, f'tfidf_vectorizer/{level}/p_vector_vector3.pkl')
df3n = vectorizer.transform(data['siblings'])
test_df3n = vectorizer.transform(test_data['siblings'])

svd = TruncatedSVD(300)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
lsa.fit(df3n)
joblib.dump(lsa, f'tfidf_vectorizer/{level}/p_lsa_vector3')
df3n = lsa.transform(df3n)
test_df3n = lsa.transform(test_df3n)
joblib.dump(df3n, f'TFIDF_Embeddings/{level}/train_siblings.pkl')
joblib.dump(test_df3n, f'TFIDF_Embeddings/{level}/test_siblings.pkl')


key_path = '/home/user/cthlx/keyword.txt'
keywordfile = open(key_path)
keywords = keywordfile.readlines()
key_list = {}
print(len(keywords))
for t in range(0,95):
    keywordss = keywords[t].replace('\n','')
    key_list[t] = keywordss.split(';')

for i in range(0,len(df)):
    list = ['0']*95
    strr = df.iloc[i]['text']
    for t in range(0,95):
        num = len(key_list[t])
        sum = 0
        if (type(strr) == 'str'):
            for k in range(0,num):
                sum+=strr.find(key_list[t][k])
            if sum+num != 0:
                list[t] = '1'
        else:
            continue
    key_str = ','.join(list)
    df.loc[i,'key'] = key_str

for i in range(0,len(test_df)):
    list = ['0']*95
    strr = test_df.iloc[i]['text']
    for t in range(0,95):
        num = len(key_list[t])
        sum = 0
        if (type(strr) == 'str'):
            for k in range(0,num):
                sum+=strr.find(key_list[t][k])
            if sum+num != 0:
                list[t] = '1'
        else:
            continue
    key_str = ','.join(list)
    test_df.loc[i,'key'] = key_str

import re
dfkey = df.iloc[:]['key'].values
list1 =[]
for item in dfkey:
    itemlist = []
    itemstr = re.findall(r'\d+',item)
    for t in itemstr:
        itemlist.append(int(t))
    itemnp = np.array(itemlist)
    list1.append(itemnp)
dfkeyn = np.array(list1)
print(np.shape(dfkeyn))

test_dfkey = test_df.iloc[:]['key'].values
list1 =[]
for item in test_dfkey:
    itemlist = []
    itemstr = re.findall(r'\d+',item)
    for t in itemstr:
        itemlist.append(int(t))
    itemnp = np.array(itemlist)
    list1.append(itemnp)
test_dfkeyn = np.array(list1)
print(np.shape(test_dfkeyn))

joblib.dump(dfkeyn, f'TFIDF_Embeddings/{level}/train_key.pkl')
joblib.dump(test_dfkeyn, f'TFIDF_Embeddings/{level}/test_key.pkl')




