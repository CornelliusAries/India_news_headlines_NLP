import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords 
from collections import Counter
import string
import re
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class SAfT_dataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer):
        self.path = path
        data = pd.read_csv(self.path)
        print(data["message to examine"])
        self.messages = torch.as_tensor(data["message to examine"])
        self.labels = torch.as_tensor(data["label (depression result)"])
        print(self.labels.shape)

    def __getitem__(self, idx):
        target = {}
        target["labels"] = torch.as_tensor(self.labels[idx], dtype=torch.int64)
        return self.messages[idx], self.labels[idx]

    def __len__(self):
        return len(self.messages)
    

def saft(data_filepath):
    # exchange with the corrupted mnist dataset
    dataset_train = SAfT_dataset(f"{data_filepath}train.csv")
    dataset_train.__len__()
    dataset_test = SAfT_dataset(f"{data_filepath}test.csv")
    dataset_test.__len__()
    train = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=0
    )
    test = torch.utils.data.DataLoader(
        dataset_test, batch_size=5000, shuffle=True, num_workers=0
    )
    return train, test

train, test = saft(r"C:\MLOPS\India_news_headlines_NLP\India_News_Headlines_NLP\data\processed\\")
