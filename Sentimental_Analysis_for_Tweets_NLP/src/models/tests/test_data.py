import sys
sys.path.append(".")

# from TweetsDataset import *
from TweetsDataset import TweetsDataset
import numpy as np
import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data import DataModule


df = pd.read_csv(f'sentiment_tweets3.csv', encoding = 'latin-1')
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

def test_DataModule_init():
    data_module = DataModule('sentiment_tweets3.csv', 16)
    assert type(data_module.data_dir) == str
    assert type(data_module.batch_size) == int
    assert data_module.batch_size > 0

def test_setup():
    data_module = DataModule('sentiment_tweets3.csv', 16)
    data_module.setup(None)
    assert type(data_module.train_dataset) == TweetsDataset
    assert type(data_module.val_dataset) == TweetsDataset

def test_train_dataloader():
    data_module = DataModule('sentiment_tweets3.csv', 16)
    data_module.setup(None)
    train_dataloader = data_module.train_dataloader()
    first_data =  next(iter(train_dataloader))
    assert type(first_data['input_ids']) == torch.Tensor
    assert first_data['input_ids'].shape == torch.Size([16, 160])
    assert type(first_data['attention_mask']) == torch.Tensor
    assert first_data['attention_mask'].shape == torch.Size([16, 160])

def test_val_dataloader():
    data_module = DataModule('sentiment_tweets3.csv', 16)
    data_module.setup(None)
    val_dataloader = data_module.val_dataloader()
    first_data =  next(iter(val_dataloader))
    assert type(first_data['input_ids']) == torch.Tensor
    assert first_data['input_ids'].shape == torch.Size([16, 160])
    assert type(first_data['attention_mask']) == torch.Tensor
    assert first_data['attention_mask'].shape == torch.Size([16, 160])