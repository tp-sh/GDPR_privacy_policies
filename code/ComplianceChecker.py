# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import joblib
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import jieba

from joblib import Parallel, delayed

os.sys.path.append("./")

from label_convertion import label_dic
from template_dic import template_dict

from loguru import logger

tfidfpaths = {
    "GDPR": {
        "title": './title_tfidfvectorizer',
        "p": './tfidfvectorizer'
    }
}

svdpaths = {
    "GDPR": {
        "title": './title_SVD',
        "p": './SVD'
    }
}

key_path = {
    "GDPR": './keyword.txt'
}

p_model_root = {
    "GDPR":"./models/pmodel"
}
title_model_root = {
    "GDPR":"./models/titlemodel"
}

key_path = './keyword.txt'
keywordfile = open(key_path)
keywords = keywordfile.readlines()
key_list = {}
for t in range(0,95):
    keywordss = keywords[t].replace('\n','')
    key_list[t] = keywordss.split(';')
    

def keywordlist(text):
    keylist = [0]*95
    
    for t in range(0,95):
        num = len(key_list[t])
        summ = 0
        if (type(text) == str):
            for k in range(0,num):
                summ+=text.find(key_list[t][k])
            if summ+num != 0:
                keylist[t] = 1
        else:
            continue
   
    return keylist

def predict(model_path, in_feature):
    """使用的模型, 输入特征, 节点号, 标题/正文"""
    model = joblib.load(model_path)
    return model.predict(in_feature)

def cleanPunc(sentence):
    """删除符号"""
    sentence = str(sentence)
    cleaned = re.sub(r'[?|!|\'|"|#]', r" ", sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r" ", cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned

def cleanPunc_ch(sentence):
    """删除符号"""
    sentence = str(sentence)
    cleaned = re.sub(r'[？|！|‘|“]', r" ", sentence)
    cleaned = re.sub(r'[?|!|\'|"|#]', r" ", sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/|<|>|;|:]', r" ", cleaned)
    cleaned = re.sub(r'[。|，|）|（|、|…|—|·|《|》|：|；|【|】]', r" ", cleaned)
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
    #NonSTOPWORDS = ['what', 'why', 'how', 'when', 'who', 'with', 'about', 'from', 'we', 'our', 'you', 'your']
    NonSTOPWORDS = []
    # 保留一些有意义的停止词
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join([ w for w in text.split() if ((w not in STOPWORDS) or (w in NonSTOPWORDS))])
    return text

def label2matrix(label_list):
    v = [0]*96
    for la in label_list:
        v[keyarr.index(la)] = 1
        lablist = la.split('.')
        if len(lablist) != 1:
            fala = lablist[0]
            v[keyarr.index(fala)] = 1
        if len(lablist) == 3:
            fala2 = lablist[0]+'.'+lablist[1]
            v[keyarr.index(fala2)] = 1
    return v
        
class ComplianceChecker:
    def __init__(self):
        self.laws = ["PIPL", "GDPR"]
        self.keyarrs = {k:label_dic[k] for k in self.laws}
        self.p_models = {}
        self.t_models = {}
        self.titles = {} # 序号-标题 {"GDPR": {序号: 标题} }
        self.models_lists = template_dict

        self.models_init()

        # 模型加载
        self.title_vectorizer = {}
        self.title_svd = {}
        self.p_vectorizer = {}
        self.p_svd = {}
        for law in self.laws:
            self.title_vectorizer[law] = joblib.load(tfidfpaths[law]["title"])
            self.title_svd[law] = joblib.load(svdpaths[law]["title"])
            self.p_vectorizer[law] = joblib.load(tfidfpaths[law]["p"])
            self.p_svd[law] = joblib.load(svdpaths[law]["p"])



    def models_init(self):
        """获取分类序号与标题关系, 加载分类器
        """
        for law in self.keyarrs:
        # for i in range(1):
            label_dict = self.keyarrs[law]
            t_models = {} # t_t_models
            p_models = {}
            titles = {}
            # p_model_root = p_model_root # To be modified
            # title_model_root = title_model_root # 改成动态
            for lineno, k in enumerate(label_dict):
                lineno += 1
                titles[lineno] = label_dict[k]
            for level, linenos in self.models_lists[law].items():
                t_models[level] = {}
                p_models[level] = {}
                for lineno in linenos.keys():
                    t_model_path = title_model_root[law] + str(lineno)
                    p_model_path = p_model_root[law] + str(lineno)
                    if (os.path.isfile(t_model_path)):
                        t_models[level][lineno] = t_model_path #joblib.load(t_model_path) #
                    if (os.path.isfile(p_model_path)):
                        p_models[level][lineno] = p_model_path  #joblib.load(p_model_path) #
            self.t_models[law] = t_models
            self.p_models[law] = p_models
            self.titles[law] = titles


    def tfidf_trans(self, textlist, law):
        text_matrix = self.p_vectorizer[law].transform(textlist)
        data_X= self.p_svd[law].transform(text_matrix)
        print(data_X.shape)
        return data_X

    def tfidf_trans_title(self, textlist, law):
        text_matrix = self.title_vectorizer[law].transform(textlist)
        data_X= self.title_svd[law].transform(text_matrix)
        print(data_X.shape)
        return data_X

    def item_preprocess(self, item, law):
        if law == "GDPR":
            item = cleanPunc(item)
            item = keepAlpha(item)
            item = text_clean(item)
        if law == "PIPL":
            item = cleanPunc_ch(item)
            item = text_clean_ch(item)

        return item
    
    def process(self, tree, law="GDPR"):
        """tree: 经过ET.parse()的xml"""
        logger.info("开始进行模型预测")
        t_models = self.t_models[law]
        p_models = self.p_models[law]

        titles = self.titles[law]
        model_list1 = self.models_lists[law][1]
        
        Level1_exist = {}
        Level1_miss = []
        model_list2 = self.models_lists[law][2]
        Level2_exist = {}
        Level2_miss = []

        root = tree.getroot()
        titletextlist = [] 
        ptextlist = []
        for child in root.iter():
            if child.tag == 'title':
                if child.text.strip() != '':
                    titletextlist.append(child.text)
            else:
                if child.text.strip() != '':
                    ptextlist.append(child.text)
        cleantlist = [self.item_preprocess(item, law) for item in titletextlist]
        cleanplist = [self.item_preprocess(item, law) for item in ptextlist]
        
        title_tfidf_matrix = self.tfidf_trans_title(cleantlist, law)
        p_tfidf_matrix = self.tfidf_trans(cleanplist, law)
        
        titlenum = 0
        pnum = 0   

        parent_map = {}
        for p in tree.iter():
            for c in p.iter():
                if c != p:
                    if c in parent_map.keys():
                        parent_map[c].append(p)

                    else:
                        parent_map[c] = [p]
        
        predicted_labels = {}
        for p in tree.iter():
            predicted_labels[p] = []

        for child in root.iter():
            if not child.text.strip(): continue
            if child.tag == 'title' and child.text == titletextlist[titlenum] and (child.attrib['category'] != ''):
                tfidf_vector = title_tfidf_matrix[titlenum]
                for sec in parent_map[child]:
                    if (sec[0].tag == 'title') and (sec[0] != child) and ('category' in sec[0].attrib.keys()) and (sec[0].text != None):
                        for item in predicted_labels[sec]:
                            parents_labels.append(item)
                for sib in parent_map[child][-2]:
                    if sib.tag == 'section' and sib[0] != child and ('category' in sib[0].attrib.keys()) and (sib[0].text != None)and (sib[0].text != ti_text):
                        for item in predicted_labels[sib]:
                            siblings_labels.append(item)
                        break
                    if sib.tag == 'section' and sib[0] == child:
                        break
                    if sib.tag == 'p' and ('category' in sib.attrib.keys())and (sib.text != None) and (sib.text != child.text):
                        for item in predicted_labels[sib]:
                            siblings_labels.append(item)
                        break
                parents_labels = list(set(parents_labels))
                siblings_labels = list(set(siblings_labels))
                parents_matrix = label2matrix(parents_labels)
                siblings_matrix = label2matrix(siblings_labels)
                keywordf = keywordlist(child.text)
                models1 = t_models[1]
                models2 = t_models[2]
                titlenum += 1
                
            elif child.text == ptextlist[pnum]: # child.tag == 'p' and 
                tfidf_vector = p_tfidf_matrix[pnum]
                for sec in parent_map[child]:
                    if (sec[0].tag == 'title') and (sec[0] != child) and ('category' in sec[0].attrib.keys()) and (sec[0].text != None):
                        for item in predicted_labels[sec]:
                            parents_labels.append(item)
                        
                for sib in parent_map[child][-2]:
                    if sib.tag == 'section' and sib[0] != child and ('category' in sib[0].attrib.keys()) and (sib[0].text != None)and (sib[0].text != ti_text):
                        for item in predicted_labels[sib]:
                            siblings_labels.append(item)
                        break
                    if sib.tag == 'section' and sib[0] == child:
                        break
                    if sib.tag == 'p' and ('category' in sib.attrib.keys())and (sib.text != None) and (sib.text != child.text):
                        for item in predicted_labels[sib]:
                            siblings_labels.append(item)
                        break
                parents_labels = list(set(parents_labels))
                siblings_labels = list(set(siblings_labels))
                parents_matrix = label2matrix(parents_labels)
                siblings_matrix = label2matrix(siblings_labels)
                keywordf = keywordlist(child.text)
                models1 = p_models[1]
                models2 = p_models[2]
                pnum += 1
            else: continue


            inputfeature = np.array([tfidf_vector,parents_matrix,siblings_matrix,keywordf])
            inputfeature = inputfeature.reshape(1,-1)
            logger.info(f"文本: {child.text.strip()[:10]}")
            for t, model in models1.items():
                model = joblib.load(model)
                out = model.predict(inputfeature)
                if out == 1:
                    Level1_exist.setdefault(t, []).append(child.text.strip() + '\n')
            for t, model in models2.items():
                model = joblib.load(model)
                out = model.predict(inputfeature)
                if out == 1:
                    Level2_exist.setdefault(t, []).append(child.text.strip() + '\n')
        Level1_exist_key = sorted(Level1_exist.keys())
        logger.info(f"一级节点命中: {Level1_exist_key}")
        Level1_miss = list(set(model_list1.keys())- set(Level1_exist.keys()))
        Level1_miss.sort()
        logger.info(f"一级节点未命中: {Level1_miss}")
        # 序号到文本映射
        Level1_exist_title = {titles[item]: Level1_exist[item] for item in Level1_exist_key}
        Level1_miss_title = {titles[item]: model_list1[item] for item in Level1_miss}

        Level2_exist_key = sorted(Level2_exist.keys())
        logger.info(f"二级节点命中: {Level2_exist_key}")
        Level2_miss = list(set(model_list2.keys())- set(Level2_exist.keys()))
        Level2_miss.sort()
        Level2_exist_title = {titles[item]: Level2_exist[item] for item in Level2_exist_key}
        Level2_miss_title = {titles[item]: model_list2[item] for item in Level2_miss}

        return Level1_exist_title, Level1_miss_title , Level2_exist_title, Level2_miss_title







            
            
