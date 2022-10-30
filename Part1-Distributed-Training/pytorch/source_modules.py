# Databricks notebook source
from pyspark.sql.functions import col

from petastorm.spark import SparkDatasetConverter, make_spark_converter

import io
import numpy as np
import torch
import torchvision
from PIL import Image
from functools import partial 
from petastorm import TransformSpec
from torchvision import transforms
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torchvision import transforms

# Create pytorch dataloader


class FlowerDataset(torch.utils.data.Dataset):
  def __init__(self, dataframe, transform=lambda arg: arg):
    self.dataframe = dataframe
    self.transform = transform
  
  def __len__(self):
    return len(self.dataframe)
  
  def __getitem__(self, idx):
    row = self.dataframe.iloc[idx]
    img = Image.open(io.BytesIO(row["content"]))
    return self.transform(img), row["label_idx"]


transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# COMMAND ----------

def build_model(num_classes):
    from torchvision import models
    import torch.nn as nn
    model = models.mobilenet_v2(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model



# COMMAND ----------

def train_one_epoch(model, criterion, optimizer, scheduler, 
                    train_dataloader, epoch, 
                    device):
  model.train()  # Set model to training mode

  # statistics
  running_loss = 0.0
  running_corrects = 0

  # Iterate over the data for one epoch.
  for input, labels in train_dataloader:
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = input.to(device), labels.type(torch.LongTensor).to(device)
    # Track history in training
    with torch.set_grad_enabled(True):
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)

      # backward + optimize
      loss.backward()
      optimizer.step()

    # statistics
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)
  
  scheduler.step()

  epoch_loss = running_loss / len(train_dataloader)
  epoch_acc = running_corrects.double() / len(train_dataloader)

  print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
  return epoch_loss, epoch_acc

def evaluate(model, criterion, val_dataloader, device, 
             metric_agg_fn=None):
  model.eval()  # Set model to evaluate mode

  # statistics
  running_loss = 0.0
  running_corrects = 0

  # Iterate over the data for one epoch.
  for input, labels in val_dataloader:
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = input.to(device), labels.type(torch.LongTensor).to(device)
    # Track history in training
    with torch.set_grad_enabled(True):
      # forward
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)

    # statistics
    running_loss += loss.item()
    running_corrects += torch.sum(preds == labels.data)
  
  # The losses are averaged across observations for each minibatch.
  epoch_loss = running_loss / len(train_dataloader)
  epoch_acc = running_corrects.double() / len(train_dataloader)
  
  # metric_agg_fn is used in the distributed training to aggregate the metrics on all workers
  if metric_agg_fn is not None:
    epoch_loss = metric_agg_fn(epoch_loss, 'avg_loss')
    epoch_acc = metric_agg_fn(epoch_acc, 'avg_acc')

  print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
  return epoch_loss, epoch_acc
