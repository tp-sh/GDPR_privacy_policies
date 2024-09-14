import os
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
from transformers import BertModel, AutoModel
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
torch.manual_seed(0)
np.random.seed(0)

model_path = 'privbert'
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load('Embedding/checkpoints/segment/epoch_9.ckpt')
state_dict_full_model = checkpoint['state_dict']

model = AutoModel.from_pretrained(model_path).to(device)
state_dict_bert = {k.replace('bert.', ''): v for k, v in state_dict_full_model.items() if 'bert.' in k}
model.load_state_dict(state_dict_bert)

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


def read_csv(file_path, attribute):
    # 返回 text, label, max_len
    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
    #df.fillna(0,inplace=True)
    #df.drop(index=(df.loc[(df['text']==0)].index))
    #missing_values_check = df.isnull().sum()
    # df.drop(index=(df.loc[(df['pa_text']==0)].index))
    print("# of rows in data = {}".format(df.shape[0]))
    print("# of columns in data = {}".format(df.shape[1]))
    data = df
    data["text"] = data["text"].str.lower()
    data["text"] = data["text"].apply(cleanPunc)
    data["text"] = data["text"].apply(keepAlpha)
    data["parents"] = data["parents"].str.lower()
    data["parents"] = data["parents"].apply(cleanPunc)
    data["parents"] = data["parents"].apply(keepAlpha)
    data["siblings"] = data["siblings"].str.lower()
    data["siblings"] = data["siblings"].apply(cleanPunc)
    data["siblings"] = data["siblings"].apply(keepAlpha)
    data["text"] = data["text"].apply(text_clean)
    data["parents"] = data["parents"].apply(text_clean)
    data["siblings"] = data["siblings"].apply(text_clean)

    data_X = data[attribute]
    data_y = data.drop(labels=["label", "text","parents","siblings","parents_matrix","siblings_matrix"], axis=1)

    return data_X.to_list(), data_y.to_numpy(), 256 # data_y.to_numpy(), 256

class MyDataset(Dataset):
    def __init__(self, texts, labels, max_length=256, model_path=model_path):
        self.all_text = texts
        self.all_label = labels
        self.max_len = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __getitem__(self, index):
        # 取出一条数据并截断长度
        text = self.all_text[index]
        label = self.all_label[index]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
            )
        token_ids = encoding['input_ids'].flatten()
        mask = encoding['attention_mask'].flatten()
        return (token_ids, mask), label, text

    def __len__(self):
        # 得到文本的长度
        return len(self.all_text)


def get_bert_embedding(file_path, save_path, attribute, model_path=model_path, device=device):

    all_text, all_label, max_len = read_csv(file_path, attribute)
    allDataset = MyDataset(all_text, all_label, max_len, model_path)
    allDataloader = DataLoader(allDataset, batch_size=4, shuffle=False, drop_last=True)

    bert_embeddings = []
    labels = []
    model.eval()
    with torch.no_grad():
        #x: 列表，每一项均为4*256的tensor，原因是batch_size=4，max_len=256，列表有两项，第一项为input_ids，第二项为attention_mask
        for x, _ , text in tqdm(allDataloader):
            input_ids, attention_mask = x[0].to(device), x[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            # embedding 4*768的tensor, 4是因为batch size=4, 768是bert的hidden_size, hidden size与pooler_output相关，原因是pooler_output是经过一层线性变换的hidden state，所以维度相同
            embedding = outputs.pooler_output
            bert_embeddings.append(embedding)
            labels.append(_)

    bert_embeddings = torch.stack(bert_embeddings).reshape([-1, 768]).to('cpu').numpy()
    labels = torch.stack(labels).reshape([-1, 96]).to('cpu').numpy()
    torch.save((bert_embeddings, labels), save_path)
    
def embedding(level, attribute):

    if level == 'document':
        file1 = ".../data/datasets/p_dataset_document_train.csv"
        dir1 = "Embedding/PrivBert_Embeddings3/document/train_"
        file2 = ".../data/datasets/p_dataset_document_test.csv"
        dir2 = "Embedding/PrivBert_Embeddings3/document/test_"
    elif level == 'segment':
        file1 = ".../data/datasets/p_dataset_segment_train.csv"
        dir1 = "Embedding/PrivBert_Embeddings3/segment/train_"
        file2 = ".../data/datasets/p_dataset_segment_test.csv"
        dir2 = "Embedding/PrivBert_Embeddings3/segment/test_"

    get_bert_embedding(file1, dir1+attribute+".pt", attribute)
    get_bert_embedding(file2, dir2+attribute+".pt", attribute)
    

if __name__ == "__main__":

    embedding('document', 'text')
    embedding('document', 'parents')
    embedding('document', 'siblings')

    embedding('segment', 'text')
    embedding('segment', 'parents')
    embedding('segment', 'siblings')
    
