import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import argparse
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
import hypertune
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import subprocess
import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch


from torch import nn, optim
# Local imports
from functions import create_data_loader
from model import DepressionClassifier
from functions import train_epoch, eval_model, loss_accuracy_plots

MODEL_FILE_NAME = 'torch.model'


def get_args():
  """Argument parser.
  Returns:
    Dictionary of arguments.
  """
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument(
      '--batch-size',
      type=int,
      default=64,
      metavar='N',
      help='input batch size for training (default: 64)')
  parser.add_argument(
      '--test-batch-size',
      type=int,
      default=1000,
      metavar='N',
      help='input batch size for testing (default: 1000)')
  parser.add_argument(
      '--epochs',
      type=int,
      default=10,
      metavar='N',
      help='number of epochs to train (default: 10)')
  parser.add_argument(
      '--lr',
      type=float,
      default=0.01,
      metavar='LR',
      help='learning rate (default: 0.01)')
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.5,
      metavar='M',
      help='SGD momentum (default: 0.5)')
  parser.add_argument(
      '--no-cuda',
      action='store_true',
      default=False,
      help='disables CUDA training')
  parser.add_argument(
      '--seed',
      type=int,
      default=1,
      metavar='S',
      help='random seed (default: 1)')
  parser.add_argument(
      '--log-interval',
      type=int,
      default=10,
      metavar='N',
      help='how many batches to wait before logging training status')
  parser.add_argument(
      '--model-dir',
      default=None,
      help='The directory to store the model')

  args = parser.parse_args()
  return args

def main():
    # Training settings
    args = get_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    #Reading raw data
    df = pd.read_csv(f'data/raw/sentiment_tweets3.csv', encoding = 'latin-1')
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    class_names = ['Not Depressed', 'Depressed']
   
    sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    encoding = tokenizer.encode_plus(sample_txt,
                                     max_length=32,
                                     add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                     return_token_type_ids=False,
                                     pad_to_max_length=True,
                                     return_attention_mask=True,
                                     return_tensors='pt')  # Return PyTorch tensors
    
    
    #Spliting datasets 
    df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
    df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
    df_val, df_test = train_test_split(df_test, test_size = 0.5, random_state = 42)
    
    
    #Creating dataloaders
    train_data_loader = create_data_loader(df_train, tokenizer, 160, args.batch_size)
    val_data_loader = create_data_loader(df_val, tokenizer,160, args.test_batch_size)
    test_data_loader = create_data_loader(df_test, tokenizer, 160, args.test_batch_size)
    
    data = next(iter(train_data_loader))
    model = DepressionClassifier(len(class_names), 'bert-base-cased')
    model = model.to(device)
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    
    F.softmax(model(input_ids, attention_mask), dim=1)
    optimizer = AdamW(model.parameters(), lr = args.lr, correct_bias = False)
    total_steps = len(train_data_loader) * args.epochs
    
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch + 1}/{args.epochs + 1}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(model,
                                            train_data_loader,    
                                            loss_fn, 
                                            optimizer, 
                                            device, 
                                            scheduler, 
                                            len(df_train))

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(model,
                                       val_data_loader,
                                       loss_fn, 
                                       device, 
                                       len(df_val))

        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        # Uses hypertune to report metrics for hyperparameter tuning.
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='my_loss',
                                                metric_value=val_loss,
                                                global_step=epoch)
    if args.model_dir:
        tmp_model_file = os.path.join('/tmp', MODEL_FILE_NAME)
        torch.save(model.state_dict(), tmp_model_file)
        subprocess.check_call(['gsutil', 'cp', tmp_model_file,
                               os.path.join(args.model_dir, MODEL_FILE_NAME)])

if __name__ == '__main__':
    main()