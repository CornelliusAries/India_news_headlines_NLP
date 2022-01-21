import sys
import os
_TEST_ROOT = os.path.dirname(__file__)
_HOME_ROOT = os.path.dirname(_TEST_ROOT)
_PROJECT_ROOT = os.path.join(_HOME_ROOT, "Sentimental_Analysis_for_Tweets_NLP/src/models")
_DATA_ROOT = os.path.join(_HOME_ROOT, "Sentimental_Analysis_for_Tweets_NLP/data/raw/sentiment_tweets3.csv")
sys.path.append(_PROJECT_ROOT)
# sys.path.append("../Sentimental_Analysis_for_Tweets_NLP/src/models")

# from TweetsDataset import *
from TweetsDataset import TweetsDataset
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from functions import create_data_loader, train_epoch, eval_model, loss_accuracy_plots
from model import DepressionClassifier
import numpy as np
from torch import nn
import pandas as pd
from collections import defaultdict
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

MAX_LEN = 160
BATCH_SIZE = 16
EPOCHS = 1
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
class_names = ['Not Depressed', 'Depressed']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(_DATA_ROOT, encoding = 'latin-1')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
loss_fn = nn.CrossEntropyLoss()

df_train = df.head(BATCH_SIZE)
df_val = df.tail(BATCH_SIZE)
train_data_loader = create_data_loader(df_train, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
model = DepressionClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)

optimizer = AdamW(model.parameters(), lr = 2e-5, correct_bias = False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,
                                                num_training_steps = total_steps)

history = defaultdict(list)
output_path = "../"


def test_train_epoch():
    train_acc, train_loss = train_epoch(model,
    train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
    assert type(train_acc) == torch.Tensor
    assert train_acc >= 0
    assert train_acc <= 1
    assert train_loss >= 0

def test_eval_model():
    val_acc, val_loss = eval_model(model,
    val_data_loader, loss_fn, device, len(df_val))

    assert type(val_acc) == torch.Tensor
    assert val_acc >= 0
    assert val_acc <= 1
    assert val_loss >= 0

def test_loss_accuracy_plots():
    loss_accuracy_plots(history, output_path)
