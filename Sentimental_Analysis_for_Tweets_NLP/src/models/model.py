from transformers import BertModel, AdamW
import torch.nn.functional as F
import numpy as np
import torch
import pytorch_lightning as pl
from matplotlib import rc
from textwrap import wrap
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DepressionClassifier(pl.LightningModule):

  def __init__(self, n_classes, pre_trained_model_name):
    super(DepressionClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(pre_trained_model_name)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,
      return_dict = False #here
    )
    output = self.drop(pooled_output)
    return self.out(output)

  def training_step(self, train_batch, batch_idx):
    losses = []
    for train_data in train_batch:
      input_ids = train_data["input_ids"].to(device)
      attention_mask = train_data["attention_mask"].to(device)
      depression = train_data["depression"].to(device)
      
      outputs = self.forward(input_ids = input_ids,
      attention_mask = attention_mask)

      loss = F.cross_entropy(outputs, depression)
      losses.append(loss.item())
    self.log('train loss', np.mean(losses))
    return loss

  def validation_step(self, val_batch, batch_idx):
    losses = []
    for val_data in val_batch:
      input_ids = val_data["input_ids"].to(device)
      attention_mask = val_data["attention_mask"].to(device)
      depression = val_data["depression"].to(device)
      
      outputs = self.forward(input_ids = input_ids,
      attention_mask = attention_mask)

      loss = F.cross_entropy(outputs, depression)
      losses.append(loss.item())
    self.log('train loss', np.mean(losses))
    return loss


  def configure_optimizers(self):
    return AdamW(self.parameters(), lr = 2e-5, correct_bias = False)