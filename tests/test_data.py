import os, sys
_TEST_ROOT = os.path.dirname(__file__)
_HOME_ROOT = os.path.dirname(_TEST_ROOT)
_PROJECT_ROOT = os.path.join(_HOME_ROOT, "Sentimental_Analysis_for_Tweets_NLP/src/models")
_DATA_ROOT = os.path.join(_HOME_ROOT, "Sentimental_Analysis_for_Tweets_NLP/data/raw/sentiment_tweets3.csv")
sys.path.append(_PROJECT_ROOT)

# sys.path.append("../Sentimental_Analysis_for_Tweets_NLP/src/models")

# from TweetsDataset import *
from TweetsDataset import TweetsDataset
from functions import create_data_loader
import numpy as np
import torch
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split


df = pd.read_csv(_DATA_ROOT, encoding = 'latin-1')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
df_val, df_test = train_test_split(df_test, test_size = 0.5, random_state = 42)

TRAIN_SIZE = 8251
VAL_SIZE = 1031
TEST_SIZE = 1032

def test_TweetsDataset():
    ds_train = TweetsDataset(
        message = df_train['message to examine'].to_numpy(),
        depression = df_train['label (depression result)'].to_numpy(),
        tokenizer=tokenizer,
        max_len=160)

    ds_val = TweetsDataset(
        message = df_val['message to examine'].to_numpy(),
        depression = df_val['label (depression result)'].to_numpy(),
        tokenizer=tokenizer,
        max_len=160)

    ds_test = TweetsDataset(
        message = df_test['message to examine'].to_numpy(),
        depression = df_test['label (depression result)'].to_numpy(),
        tokenizer=tokenizer,
        max_len=160)

    assert len(ds_train) == TRAIN_SIZE
    assert len(ds_val) == VAL_SIZE
    assert len(ds_test) == TEST_SIZE
    

def test_create_dataloader():
    data_loader = create_data_loader(df, tokenizer, max_len=160, batch_size=16)
    first_data =  next(iter(data_loader))
    assert type(first_data['input_ids']) == torch.Tensor
    assert first_data['input_ids'].shape == torch.Size([16, 160])
    assert type(first_data['attention_mask']) == torch.Tensor
    assert first_data['attention_mask'].shape == torch.Size([16, 160])

