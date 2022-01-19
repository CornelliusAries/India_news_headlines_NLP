from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

#Local imports
from TweetsDataset import TweetsDataset

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = TweetsDataset(
    message = df['message to examine'].to_numpy(),
    depression = df['label (depression result)'].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size = batch_size,
    num_workers = 9
  )
  
def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    depression = d["depression"].to(device)

    outputs = model(
      input_ids = input_ids,
      attention_mask = attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, depression)

    correct_predictions += torch.sum(preds == depression)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      depression = d["depression"].to(device)

      outputs = model(
        input_ids = input_ids,
        attention_mask = attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, depression)

      correct_predictions += torch.sum(preds == depression)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

def loss_accuracy_plots(history, figures_output_filepath):
    plt.figure(1)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.xlabel("Epochs [-]")
    plt.ylabel("Loss [-]")
    plt.legend(['Training loss','Validation loss'])
    plt.grid()
    plt.savefig(f"{figures_output_filepath}/Training_losses_plot.png")
    plt.figure(2)
    plt.plot(history['train_acc'])
    plt.plot(history['val_acc'])
    plt.xlabel("Epochs [-]")
    plt.ylabel("Loss [-]")
    plt.legend(['Training accuracy','Validation accuracy'])
    plt.grid()
    plt.savefig(f"{figures_output_filepath}/Training_accuracies_plot.png")