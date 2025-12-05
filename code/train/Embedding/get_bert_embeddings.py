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
import argparse

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--level', type=str, default='segment') #segment
parser.add_argument('--fold', type=str, default='1')
args = parser.parse_args()

level = args.level
fold = args.fold

model_path = '/data/data1/cyx/privbert'
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(f'cross_validation/checkpoints/{level}/fold_{fold}/epoch_9.ckpt')
state_dict_full_model = checkpoint['state_dict']

model = AutoModel.from_pretrained(model_path).to(device)
state_dict_bert = {k.replace('bert.', ''): v for k, v in state_dict_full_model.items() if 'bert.' in k}
model.load_state_dict(state_dict_bert)

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


def text_clean(text):
    lem = WordNetLemmatizer()  # 词性还原
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    #NonSTOPWORDS = ['what', 'why', 'how', 'when', 'who', 'with', 'about', 'from', 'we', 'our', 'you', 'your']
    NonSTOPWORDS = []
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join([ w for w in text.split() if ((w not in STOPWORDS) or (w in NonSTOPWORDS))])
    return text


def read_csv(file_path, attribute):
    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
    df.fillna(0,inplace=True)
    df.drop(index=(df.loc[(df['text']==0)].index))
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
        return len(self.all_text)


def get_bert_embedding(file_path, save_path, attribute, model_path=model_path, device=device):

    all_text, all_label, max_len = read_csv(file_path, attribute)
    allDataset = MyDataset(all_text, all_label, max_len, model_path)
    allDataloader = DataLoader(allDataset, batch_size=4, shuffle=False, drop_last=True)

    bert_embeddings = []
    labels = []
    model.eval()
    with torch.no_grad():
        for x, _ , text in tqdm(allDataloader):
            input_ids, attention_mask = x[0].to(device), x[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            embedding = outputs.pooler_output
            bert_embeddings.append(embedding)
            labels.append(_)

    bert_embeddings = torch.stack(bert_embeddings).reshape([-1, 768]).to('cpu').numpy()
    labels = torch.stack(labels).reshape([-1, 96]).to('cpu').numpy()
    torch.save((bert_embeddings, labels), save_path)
    
def embedding(level, attribute):
    file1 = f"/data/data1/cyx/cross_validation/{level}/fold_{fold}_train.csv"
    file2 = f"/data/data1/cyx/cross_validation/{level}/fold_{fold}_test.csv"
    dir1 = f"cross_validation/PrivBert_Embeddings_fold{fold}/{level}/train_"
    dir2 = f"cross_validation/PrivBert_Embeddings_fold{fold}/{level}/test_"
    
    get_bert_embedding(file1, dir1+attribute+".pt", attribute)
    get_bert_embedding(file2, dir2+attribute+".pt", attribute)
    

if __name__ == "__main__":

    embedding(level, 'text')
    embedding(level, 'parents')
    embedding(level, 'siblings')
    
