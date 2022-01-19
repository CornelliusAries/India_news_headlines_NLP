from transformers import BertModel, AdamW
import torch.nn.functional as F
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
    input_ids = train_batch["input_ids"].to(device)
    attention_mask = train_batch["attention_mask"].to(device)
    labels = train_batch["depression"].to(device)
    outputs = self.forward(input_ids, attention_mask)
    loss = F.cross_entropy(outputs, labels)
    self.log('train loss', loss)

    return loss

  def validation_step(self, val_batch, batch_idx):
    input_ids = val_batch["input_ids"].to(device)
    attention_mask = val_batch["attention_mask"].to(device)
    labels = val_batch["depression"].to(device)
    outputs = self.forward(input_ids, attention_mask)
    loss = F.cross_entropy(outputs, labels)
    self.log('train loss', loss)


  def configure_optimizers(self):
    return AdamW(self.parameters(), lr = 2e-5, correct_bias = False)