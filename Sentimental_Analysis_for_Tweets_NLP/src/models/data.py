import pytorch_lightning as pl
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
from TweetsDataset import TweetsDataset


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_LEN = 160
# BATCH_SIZE = 16
EPOCHS = 10
    
# df = pd.read_csv('sentiment_tweets3.csv', encoding = 'latin-1')
# df = pd.read_csv('data/raw/sentiment_tweets3.csv', encoding = 'latin-1')
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
# df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
# df_val, df_test = train_test_split(df_test, test_size = 0.5, random_state = 42)


class DataModule(pl.LightningDataModule):
  def __init__(self, data_dir, batch_size):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size

  def setup(self, stage):
    df = pd.read_csv(self.data_dir, encoding = 'latin-1')
    df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
    df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
    df_val, df_test = train_test_split(df_test, test_size = 0.5, random_state = 42)
    self.train_dataset = TweetsDataset(
        message = df_train['message to examine'].to_numpy(),
        depression = df_train['label (depression result)'].to_numpy(),
        tokenizer=tokenizer,
        max_len= MAX_LEN)
    
    self.val_dataset = TweetsDataset(
        message = df_val['message to examine'].to_numpy(),
        depression = df_val['label (depression result)'].to_numpy(),
        tokenizer=tokenizer,
        max_len= MAX_LEN)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers = 0)

  def val_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers = 0)