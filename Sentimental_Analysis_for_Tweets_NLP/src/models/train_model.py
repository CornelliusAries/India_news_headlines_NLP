import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd

from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv

from torch import nn, optim
# Local imports
from functions import create_data_loader
from model import DepressionClassifier
from functions import train_epoch, eval_model, loss_accuracy_plots


@click.command()
@click.argument("model_output_filepath", type=click.Path())
@click.argument("data_filepath", type=click.Path(exists=True))
@click.argument("figures_output_filepath", type=click.Path())
def main(model_output_filepath="models", data_filepath="data/processed", figures_output_filepath="reports/figures"):
    """Trains model and saves it for evaluation"""
    
    #HYPER PARAMETERS
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    MAX_LEN = 160
    BATCH_SIZE = 16
    EPOCHS = 5
    
    #Reading raw data
    df = pd.read_csv(f'{data_filepath}/sentiment_tweets3.csv', encoding = 'latin-1')
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    class_names = ['Not Depressed', 'Depressed']
   
    sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
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
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
    
    data = next(iter(train_data_loader))
    model = DepressionClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
    model = model.to(device)
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    
    F.softmax(model(input_ids, attention_mask), dim=1)
    optimizer = AdamW(model.parameters(), lr = 2e-5, correct_bias = False)
    total_steps = len(train_data_loader) * EPOCHS
    
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
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
    torch.save(model.state_dict(), model_output_filepath + f"/{PRE_TRAINED_MODEL_NAME}_depression_classifier.pth")
    
    loss_accuracy_plots(history, figures_output_filepath)
        
        
if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()