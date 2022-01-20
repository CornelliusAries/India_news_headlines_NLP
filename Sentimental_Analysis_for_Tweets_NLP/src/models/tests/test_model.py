import sys
sys.path.append(".")

# from TweetsDataset import *
from TweetsDataset import TweetsDataset
import transformers
from transformers import BertModel
from functions import create_data_loader
from model import DepressionClassifier
import numpy as np
from torch import nn
import pandas as pd
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

MAX_LEN = 160
BATCH_SIZE = 16
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
class_names = ['Not Depressed', 'Depressed']

df = pd.read_csv(f'../../data/raw/sentiment_tweets3.csv', encoding = 'latin-1')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
data_loader = create_data_loader(df_train, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
first_data =  next(iter(data_loader))
input_ids = first_data['input_ids']
attention_mask = first_data['attention_mask']

def test_DepressionClassifier_init():
    model = DepressionClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
    assert type(model.bert) == transformers.models.bert.modeling_bert.BertModel
    assert type(model.drop) == nn.Dropout
    assert type(model.out) == nn.Linear

def test_DepressionClassifier_forward():
    model = DepressionClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
    out = model(input_ids, attention_mask)
    assert first_data['input_ids'].shape == torch.Size([BATCH_SIZE, MAX_LEN])
    assert first_data['attention_mask'].shape == torch.Size([BATCH_SIZE, MAX_LEN])
    assert out.ndim == 2
    assert out.shape == torch.Size([BATCH_SIZE, 2])
