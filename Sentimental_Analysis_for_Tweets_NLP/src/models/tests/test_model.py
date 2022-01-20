import sys
sys.path.append(".")

# from TweetsDataset import *
from TweetsDataset import TweetsDataset
import transformers
from transformers import BertModel

from model import DepressionClassifier
import numpy as np
from torch import nn
import pandas as pd
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from data import DataModule

MAX_LEN = 160
BATCH_SIZE = 16
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
class_names = ['Not Depressed', 'Depressed']


data_module = DataModule('mini.csv', BATCH_SIZE)
data_module.setup(None)
train_dataloader = data_module.train_dataloader()
val_dataloader = data_module.val_dataloader()
first_data =  next(iter(train_dataloader))
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
    assert first_data['input_ids'].shape == torch.Size([12, MAX_LEN])
    assert first_data['attention_mask'].shape == torch.Size([12, MAX_LEN])
    assert out.ndim == 2
    assert out.shape == torch.Size([12, 2])

def test_training_step():
    model = DepressionClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
    train_loss = model.training_step(train_dataloader, 0)
    assert train_loss >= 0

def test_validation_step():
    model = DepressionClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
    val_loss = model.validation_step(val_dataloader, 0)
    assert val_loss >= 0

def test_configure_optimizers():
    model = DepressionClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
    optimizer = model.configure_optimizers()
    assert type(optimizer) == transformers.optimization.AdamW

