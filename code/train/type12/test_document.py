# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import numpy as np
from train import PrivacyTagger
from transformers import AutoTokenizer, AutoModel
import torch
import sys

sys.path.append('../')
from label_convertion import label_dic

device = torch.device('cuda:0')
keyarr = list(label_dic.keys())

exceptlist = [41, 27, 10, 118, 5, 133, 15, 101, 49, 114, 31, 139, 115, 148, 30, 91, 33, 136, 128, 144, 146, 70, 24, 79, 13, 25, 135, 29, 56, 78]

modellist_level1 =  ['1', '22', '38', '39', '41', '47', '54', '64', '65', '67', '85', '86', '90', '93']
modellist_level2_1 =  ['2', '4']
modellist_level2_2 =  ['23', '27', '28', '29', '30', '31', '32', '34']
modellist_level2_3 =  ['48', '49']
modellist_level2_4 =  ['55', '60', '62', '63']
modellist_level2_5 =  ['87', '88', '89']
modellist_level2_6 =  ['91', '92']

modelP_level1 = PrivacyTagger.load_from_checkpoint('type6/checkpoints/document/1/epoch_9.ckpt', n_classes=14, logfile_path='').to(device)
#modelP = AutoModel.from_pretrained("mukund/privbert")
modelT_level1 = modelP_level1.to(device)

modelP_level2_1 = PrivacyTagger.load_from_checkpoint('type6/checkpoints/document/2_1/epoch_8.ckpt', n_classes=2, logfile_path='').to(device)
#modelP = AutoModel.from_pretrained("mukund/privbert")
modelT_level2_1 = modelP_level2_1.to(device)

modelP_level2_2 = PrivacyTagger.load_from_checkpoint('type6/checkpoints/document/2_2/epoch_9.ckpt', n_classes=8, logfile_path='').to(device)
#modelP = AutoModel.from_pretrained("mukund/privbert")
modelT_level2_2 = modelP_level2_2.to(device)

modelP_level2_3 = PrivacyTagger.load_from_checkpoint('type6/checkpoints/document/2_3/epoch_7.ckpt', n_classes=2, logfile_path='').to(device)
#modelP = AutoModel.from_pretrained("mukund/privbert")
modelT_level2_3 = modelP_level2_3.to(device)

modelP_level2_4 = PrivacyTagger.load_from_checkpoint('type6/checkpoints/document/2_4/epoch_9.ckpt', n_classes=4, logfile_path='').to(device)
#modelP = AutoModel.from_pretrained("mukund/privbert")
modelT_level2_4 = modelP_level2_4.to(device)

modelP_level2_5 = PrivacyTagger.load_from_checkpoint('type6/checkpoints/document/2_5/epoch_6.ckpt', n_classes=3, logfile_path='').to(device)
#modelP = AutoModel.from_pretrained("mukund/privbert")
modelT_level2_5 = modelP_level2_5.to(device)

modelP_level2_6 = PrivacyTagger.load_from_checkpoint('type6/checkpoints/document/2_6/epoch_9.ckpt', n_classes=2, logfile_path='').to(device)
#modelP = AutoModel.from_pretrained("mukund/privbert")
modelT_level2_6 = modelP_level2_6.to(device)

tokenizer = AutoTokenizer.from_pretrained("privbert")

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
    #NonSTOPWORDS = ['what', 'why', 'how', 'when', 'who', 'with', 'about', 'from', 'we', 'our', 'you', 'your']
    NonSTOPWORDS = []
    # 保留一些有意义的停止词
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join([ w for w in text.split() if ((w not in STOPWORDS) or (w in NonSTOPWORDS))])
    return text


def term_identification(number):
    tTP = [0]*35
    tTN = [0]*35
    tFP = [0]*35
    tFN = [0]*35
    
    pTP = [0]*35
    pTN = [0]*35
    pFP = [0]*35
    pFN = [0]*35
    try:
        tree = ET.parse('GoPPC-150/' + str(number) + '.xml')
    except:
        return [tTP, tTN, tFP, tFN, pTP, pTN, pFP, pFN]
    root = tree.getroot()
    ptextlist = []
    for child in root.iter():
        if child.tag == 'paragraph':
            if child.text != '' and child.text!=None:
                ptextlist.append(child.text)
    
    cleanplist = []
     
    for item in ptextlist:
        item2 = cleanPunc(item)
        item2 = keepAlpha(item2)
        item2 = text_clean(item2)
        cleanplist.append(item2)
    
    parent_map = {}
    for p in tree.iter():
        for c in p.iter():
            if c != p:
                if c in parent_map.keys():
                    parent_map[c].append(p)

                else:
                    parent_map[c] = [p]
                    
    for child in root.iter():
          
        if child.tag == 'paragraph':
            ti_text = child.text
            if ti_text != '' and ti_text!=None:
                ti_text = cleanPunc(ti_text)
                ti_text = keepAlpha(ti_text)
                ti_text = text_clean(ti_text)
                
                parent_text = ''
                sibling_text = ''
                ptextlist.append(child.text)

                for sec in parent_map[child]:
                    if (sec[0].tag == 'title') and (sec[0] != child) and ('category' in sec[0].attrib.keys()) and (sec[0].text != None):
                        parent_text+=sec[0].text
                try:        
                    for sib in parent_map[child][-1]:
                        if sib.tag == 'segment' and sib[0] != child and ('category' in sib[0].attrib.keys()) and (sib[0].text != None)and (sib[0].text != ti_text):
                            sibling_text = sib[0].text
                            break
                        if sib.tag == 'paragraph' and ('category' in sib.attrib.keys())and (sib.text != None) and (sib.text != ti_text):
                            sibling_text = sib.text
                            break
                except:
                    sibling_text = ''

                parent_text = cleanPunc(parent_text)
                parent_text = keepAlpha(parent_text)
                parent_text = text_clean(parent_text)

                sibling_text = cleanPunc(sibling_text)
                sibling_text = keepAlpha(sibling_text)
                sibling_text = text_clean(sibling_text)

                try:
                    labels = child.attrib['category'].split(';')[1]
                    labelss = labels.split(',')
                    finalLabel = []
                    for tag in labelss:
                        finalLabel.append(tag)
                        tags = tag.split('.')
                        if len(tags) >= 2:
                            finalLabel.append(tags[0])
                        if len(tags) == 3:
                            finalLabel.append(tags[0]+'.'+tags[1])
                except:
                    continue

                inputs1 = tokenizer.encode_plus(
                    ti_text,
                    add_special_tokens=True,
                    max_length=256,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                    ).to(device)
                
                inputs2 = tokenizer.encode_plus(
                    parent_text,
                    add_special_tokens=True,
                    max_length=256,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                    ).to(device)
                
                inputs3 = tokenizer.encode_plus(
                    sibling_text,
                    add_special_tokens=True,
                    max_length=256,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                    ).to(device)
                
                _, output = modelP_level1(inputs1['input_ids'], inputs1['attention_mask'], inputs2['input_ids'], inputs2['attention_mask'], inputs3['input_ids'], inputs3['attention_mask'])
                _, output2_1 = modelP_level2_1(inputs1['input_ids'], inputs1['attention_mask'], inputs2['input_ids'], inputs2['attention_mask'], inputs3['input_ids'], inputs3['attention_mask'])
                _, output2_2 = modelP_level2_2(inputs1['input_ids'], inputs1['attention_mask'], inputs2['input_ids'], inputs2['attention_mask'], inputs3['input_ids'], inputs3['attention_mask'])
                _, output2_3 = modelP_level2_3(inputs1['input_ids'], inputs1['attention_mask'], inputs2['input_ids'], inputs2['attention_mask'], inputs3['input_ids'], inputs3['attention_mask'])
                _, output2_4 = modelP_level2_4(inputs1['input_ids'], inputs1['attention_mask'], inputs2['input_ids'], inputs2['attention_mask'], inputs3['input_ids'], inputs3['attention_mask'])
                _, output2_5 = modelP_level2_5(inputs1['input_ids'], inputs1['attention_mask'], inputs2['input_ids'], inputs2['attention_mask'], inputs3['input_ids'], inputs3['attention_mask'])
                _, output2_6 = modelP_level2_6(inputs1['input_ids'], inputs1['attention_mask'], inputs2['input_ids'], inputs2['attention_mask'], inputs3['input_ids'], inputs3['attention_mask'])
                for t in range (0,14):
                    numm = torch.tensor(0.5)
                    out = (output[0][t]+numm).int()
                    tag = keyarr[int(modellist_level1[t])-1]
                    if out == 1 and (tag in finalLabel):
                        pTP[t] +=1
                    elif out == 1 and (tag not in finalLabel):
                        pFP[t] +=1
                    elif out == 0 and (tag in finalLabel):
                        pFN[t] +=1
                    elif out == 0 and (tag not in finalLabel):
                        pTN[t] +=1
                for t in range (0,2):
                    numm = torch.tensor(0.5)
                    out = (output2_1[0][t]+numm).int()
                    tag = keyarr[int(modellist_level2_1[t])-1]
                    # 要判断父节点是否被预测为真,只有父节点和子节点同时预测为真，output才为真
                    out = (out == 1 and (output[0][0]+numm).int() == 1)
                    if out == 1 and (tag in finalLabel):
                        pTP[t+14] +=1
                    elif out == 1 and (tag not in finalLabel) :
                        pFP[t+14] +=1
                    elif out == 0 and (tag in finalLabel):
                        pFN[t+14] +=1
                    elif out == 0 and (tag not in finalLabel):
                        pTN[t+14] +=1
                for t in range (0,8):
                    numm = torch.tensor(0.5)
                    out = (output2_2[0][t]+numm).int()
                    tag = keyarr[int(modellist_level2_2[t])-1]
                    # 要判断父节点是否被预测为真,只有父节点和子节点同时预测为真，output才为真
                    out = (out == 1 and (output[0][1]+numm).int() == 1)
                    if out == 1 and (tag in finalLabel):
                        pTP[t+16] +=1
                    elif out == 1 and (tag not in finalLabel) :
                        pFP[t+16] +=1
                    elif out == 0 and (tag in finalLabel):
                        pFN[t+16] +=1
                    elif out == 0 and (tag not in finalLabel):
                        pTN[t+16] +=1
                for t in range (0,2):
                    numm = torch.tensor(0.5)
                    out = (output2_3[0][t]+numm).int()
                    tag = keyarr[int(modellist_level2_3[t])-1]
                    # 要判断父节点是否被预测为真,只有父节点和子节点同时预测为真，output才为真
                    out = (out == 1 and (output[0][5]+numm).int() == 1)
                    if out == 1 and (tag in finalLabel):
                        pTP[t+24] +=1
                    elif out == 1 and (tag not in finalLabel) :
                        pFP[t+24] +=1
                    elif out == 0 and (tag in finalLabel):
                        pFN[t+24] +=1
                    elif out == 0 and (tag not in finalLabel):
                        pTN[t+24] +=1
                for t in range (0,4):
                    numm = torch.tensor(0.5)
                    out = (output2_4[0][t]+numm).int()
                    tag = keyarr[int(modellist_level2_4[t])-1]
                    # 要判断父节点是否被预测为真,只有父节点和子节点同时预测为真，output才为真
                    out = (out == 1 and (output[0][6]+numm).int() == 1)
                    if out == 1 and (tag in finalLabel):
                        pTP[t+26] +=1
                    elif out == 1 and (tag not in finalLabel) :
                        pFP[t+26] +=1
                    elif out == 0 and (tag in finalLabel):
                        pFN[t+26] +=1
                    elif out == 0 and (tag not in finalLabel):
                        pTN[t+26] +=1
                for t in range (0,3):
                    numm = torch.tensor(0.5)
                    out = (output2_5[0][t]+numm).int()
                    tag = keyarr[int(modellist_level2_5[t])-1]
                    # 要判断父节点是否被预测为真,只有父节点和子节点同时预测为真，output才为真
                    out = (out == 1 and (output[0][11]+numm).int() == 1)
                    if out == 1 and (tag in finalLabel):
                        pTP[t+30] +=1
                    elif out == 1 and (tag not in finalLabel) :
                        pFP[t+30] +=1
                    elif out == 0 and (tag in finalLabel):
                        pFN[t+30] +=1
                    elif out == 0 and (tag not in finalLabel):
                        pTN[t+30] +=1
                for t in range (0,2):
                    numm = torch.tensor(0.5)
                    out = (output2_6[0][t]+numm).int()
                    tag = keyarr[int(modellist_level2_6[t])-1]
                    # 要判断父节点是否被预测为真,只有父节点和子节点同时预测为真，output才为真
                    out = (out == 1 and (output[0][12]+numm).int() == 1)
                    if out == 1 and (tag in finalLabel):
                        pTP[t+33] +=1
                    elif out == 1 and (tag not in finalLabel) :
                        pFP[t+33] +=1
                    elif out == 0 and (tag in finalLabel):
                        pFN[t+33] +=1
                    elif out == 0 and (tag not in finalLabel):
                        pTN[t+33] +=1
                
        # print(predicted_labels)
    metrics = []
    metrics.append(tTP)
    metrics.append(tFP)
    metrics.append(tFN)
    metrics.append(tTN)
    metrics.append(pTP)
    metrics.append(pFP)
    metrics.append(pFN)
    metrics.append(pTN)
    print(str(number)+ '   \n')
    #print(metrics)
    return metrics

tTPs = [0]*35
tFPs = [0]*35
tFNs = [0]*35
tTNs = [0]*35
    
pTPs = [0]*35
pFPs = [0]*35
pFNs = [0]*35
pTNs = [0]*35


for item in exceptlist:
    metrics = term_identification(item)
 
    for i in range(0,35):
        tTPs[i]+=metrics[0][i]
        tFPs[i]+=metrics[1][i]
        tFNs[i]+=metrics[2][i]
        tTNs[i]+=metrics[3][i]
        
        pTPs[i]+=metrics[4][i]
        pFPs[i]+=metrics[5][i]
        pFNs[i]+=metrics[6][i]
        pTNs[i]+=metrics[7][i]
        
tPr = []
tRe = []
tF1 = []
pPr = []
pRe = []
pF1 = []
    
for i in range(0,35):
    try:
        tpr = (tTPs[i]/(tTPs[i]+tFPs[i]))
    except:
        tpr = 0
    try:
        tre = (tTPs[i]/(tTPs[i]+tFNs[i]))
    except:
        tre = 0
    tPr.append(tpr)
    tRe.append(tre)
    try:
        tF1.append((2*tpr*tre/(tpr+tre)))
    except:
        tF1.append(0)
    try:
        ppr = (pTPs[i]/(pTPs[i]+pFPs[i]))
    except:
        ppr = 0
    try:
        pre = (pTPs[i]/(pTPs[i]+pFNs[i]))
    except:
        pre = 0        
    pPr.append(ppr)
    pRe.append(pre)
    try:
        pF1.append((2*ppr*pre/(ppr+pre)))
    except:
        pF1.append(0)
reslog = open('type6/logfiles/document/result.txt', 'a')
reslog.write(str(pTPs)+'\n')
reslog.write(str(pFPs)+'\n')
reslog.write(str(pTNs)+'\n')
reslog.write(str(pFNs)+'\n')
reslog.write('           \n')
reslog.write(str(pPr)+'\n')
reslog.write(str(pRe)+'\n')
reslog.write(str(pF1)+'\n')
reslog.write('average F1: '+str(np.mean(pF1))+'\n')
reslog.close()
print(pF1)
            
            
            
            


