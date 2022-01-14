import torch 
import transformers
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import (
    XLNetTokenizer, 
    TFXLNetForSequenceClassification, 
    BertModel, 
    BertTokenizer, 
)

# Variables
class_names = ['positive', 'negative']

# Function to convert labels to number
def sentiment2label(sentiment):
    if sentiment == 'positive':
        return 1
    else:
        return 0

# Function to clean text(remove tagged entities, hyperlinks, emojis)
def clean_text(text):
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ',  text)
    text = re.sub(r" +", ' ', text)
    return text
 

# Loading data and preprocessing
df = pd.read_csv(r'C:\SCHOOL\MLOps\India_news_headlines_NLP\India_News_Headlines_NLP\data\raw\india-news-headlines.csv', encoding = 'latin-1')
df = df[["headline_text"]]
df = df.dropna()
#print(df.head)


# DataLoader for train/test



# Loading pretrained models
# XLNet 
XLNet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', num_labels = 2)
XLNet_model = TFXLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
"""
inputs = XLNet_tokenizer.tokenize(df.columns[0])
encodings = XLNet_tokenizer.encode_plus(
    input_txt, 
    add_special_tokens=True,
    max_length=16,
    return_tensors='pt',
    return_token_type_ids=False, 
    return_attention_mask=True, 
    pad_to_max_length=True
)
"""

# Bert
Bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
Bert_model = BertModel.from_pretrained('bert-base-cased')

"""
bert_tokens = Bert_tokenizer.tokenize(df.columns[0])
bert_token_ids = Bert_tokenizer.convert_tokens_to_ids(bert_tokens)
"""