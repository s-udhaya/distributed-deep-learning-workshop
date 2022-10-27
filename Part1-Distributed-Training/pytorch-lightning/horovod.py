# Databricks notebook source
# MAGIC %run ../00_setup 

# COMMAND ----------

# MAGIC %run ./source_modules

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

NUM_DEVICES = 2
BATCH_SIZE = 64

READER_POOL_TYPE = "dummy"
WORKERS_COUNT = 1

# COMMAND ----------

from petastorm.spark import SparkDatasetConverter

dbutils.fs.rm("file:///dbfs/tmp/petastorm/cache2", True)

Data_Directory = '/databricks-datasets/flowers/delta'

CACHE_DIR = "file:///dbfs/tmp/petastorm/cache2"


spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, CACHE_DIR)

flowers_df, train_converter, val_converter = prepare_data(data_dir=Data_Directory, 
                                                          num_devices=NUM_DEVICES)


# COMMAND ----------

def data_loader(data_converter, transform_spec,
                num_epochs,
                workers_count,
                cur_shard,
                shard_count,
              reader_pool_type,
                batch_size):
  dataloader = data_converter.make_torch_dataloader(
                transform_spec=transform_spec,
                num_epochs=num_epochs,
                workers_count=workers_count,
                cur_shard=cur_shard,
                shard_count=shard_count,
              reader_pool_type=reader_pool_type,
                batch_size=batch_size)
  return dataloader

# COMMAND ----------

import pytorch_lightning as pl
from torchvision import models
import torch.nn.functional as F
import torchmetrics.functional as FM
import torch
import logging
import datetime as dt

class LitClassificationModel(pl.LightningModule):
  """
  
  Our main model class
  
  
  """
  
  def __init__(self, class_count: int, learning_rate:float, momentum:float=0.9, logging_level=logging.INFO,
              device_id:int=0, device_count:int=1, family:str='mobilenet', use_horovod: bool = False):
    
    super().__init__()
    self.learn_rate = learning_rate
    self.momentum = momentum
    self.model = self.get_model(class_count, learning_rate, family)
    self.state = {"epochs": 0}
    self.logging_level = logging_level
    self.device_id = device_id
    self.device_count = device_count
    self.family = family
  
  def get_model(self, class_count, lr, family):
    """
    
    This is the function that initialises our model.
    If we wanted to use other prebuilt model libraries like timm we would put that model here
    
    """
    
    if family == 'mobilenet':
      model = models.mobilenet_v2(pretrained=True)
    elif family == 'resnext':
      model = models.resnext50_32x4d(pretrained=True)
    
    # Freeze parameters in the feature extraction layers and replace the last layer
    for param in model.parameters():
      param.requires_grad = False

    # New modules have `requires_grad = True` by default
    if family == 'mobilenet':
      model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, class_count)
    elif family == 'resnext':
      model.fc = torch.nn.Linear(model.fc.in_features, class_count)
    
    
    return model
  
  def configure_optimizers(self):
    
    if self.family == 'mobilenet':
      params = self.model.classifier[1].parameters()
    elif self.family == 'resnext':
      params = self.model.fc.parameters()
    
    optimizer = torch.optim.SGD(params, lr=self.learn_rate, momentum=self.momentum)
    if self.use_horovod:
      # Broadcast initial parameters so all workers start with the same parameters.
      hvd.broadcast_parameters(model.state_dict(), root_rank=0)
      hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    return optimizer
  
  def forward(self, inputs):
    outputs = self.model(inputs)
    
    return outputs
  
  def training_step(self, batch, batch_idx):
    X, y = batch["features"], batch["label"]
    pred = self(X)
    loss = F.cross_entropy(pred, y)
    
    # Choosing to use step loss as a metric
    self.log("train_loss", loss, prog_bar=True)
    
    if self.logging_level == logging.DEBUG:
      if batch_idx == 0:
        print(f" - [{self.device_id}] training batch size: {y.shape[0]}")
      print(f" - [{self.device_id}] training batch: {batch_idx}, loss: {loss}")
      
    return loss
  
  def on_train_epoch_start(self):
    # No need to re-load data here as `train_dataloader` will be called on each epoch
    if self.logging_level in (logging.DEBUG, logging.INFO):
      print(f"++ [{self.device_id}] Epoch: {self.state['epochs']}")
    self.state["epochs"] += 1
    
  def validation_step(self, batch, batch_idx):
    X, y = batch["features"], batch["label"]
    pred = self(X)
    loss = F.cross_entropy(pred, y)
    acc = FM.accuracy(pred, y)

    # Roll validation up to epoch level
    self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
    self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
    
    if self.logging_level == logging.DEBUG:
      print(f" - [{self.device_id}] val batch: {batch_idx}, size: {y.shape[0]}, loss: {loss}, acc: {acc}")

    return {"loss": loss, "acc": acc}

# COMMAND ----------

from math import ceil
import mlflow
from functools import partial


train_dataloader_callable = partial(data_loader, train_converter)
val_dataloader_callable = partial(data_loader, val_converter)

datamodule = FlowersDataModule(train_dataloader_callable=train_dataloader_callable, 
                               val_dataloader_callable=val_dataloader_callable, batch_size=BATCH_SIZE, workers_count=WORKERS_COUNT, reader_pool_type=READER_POOL_TYPE, label_column="label")

model = LitClassificationModel(class_count=5, learning_rate=1e-5, label_column="label")

train_steps_per_epoch = ceil(len(train_converter) //  BATCH_SIZE)
val_steps_per_epoch = ceil(len(val_converter) //  BATCH_SIZE)

default_dir = f'/dbfs/Users/{username}/tmp/lightning'

prepare_mlflow_experiment(username, DATABRICKS_HOST, DATABRICKS_TOKEN, "pytorch-lightning-petastorm-dataframe-load")

train(model, datamodule, gpus=1, default_dir=default_dir, batch_size=BATCH_SIZE, train_steps_per_epoch=train_steps_per_epoch, val_steps_per_epoch=val_steps_per_epoch, workers_count=WORKERS_COUNT, 
      reader_pool_type=READER_POOL_TYPE, max_epochs=1000)

# COMMAND ----------

from pytorch_lightning.utilities import _HOROVOD_AVAILABLE
if _HOROVOD_AVAILABLE:
  print("Horovod is available")
  import horovod
  import horovod.torch as hvd
  from pytorch_lightning.plugins.training_type.horovod import HorovodPlugin
  print(f"Horovod: {horovod.__version__}")

# COMMAND ----------

import mlflow
from functools import partial

def train_hvd():
  from math import ceil
  import sys
  print("inside train")
  print(sys.path)

  print("imported src")
  print(dir(LitClassificationModel))

  hvd.init()
  import sys
  train_dataloader_callable = partial(data_loader, train_converter)
  val_dataloader_callable = partial(data_loader, val_converter)
  print("importing src")

  # mlflow workaround to ensure that horovod subprocesses can find and connect to mlflow
  mlflow.set_tracking_uri("databricks")
  experiment = prepare_mlflow_experiment(username, DATABRICKS_HOST, DATABRICKS_TOKEN, "pytorch-lightning-petastorm-dataframe-load")
  
  ## we pass the experiment id over to the workers
  mlflow_experiment_id = experiment.experiment_id
  train_steps_per_epoch = ceil(len(train_converter) //  (BATCH_SIZE * NUM_DEVICES))
  val_steps_per_epoch = ceil(len(val_converter) //  (BATCH_SIZE * NUM_DEVICES))  
  hvd_model = LitClassificationModel(class_count=5, learning_rate=1e-5*hvd.size(), 
                                     device_id=hvd.rank(), device_count=hvd.size(), label_column="label")
  
  hvd_datamodule = FlowersDataModule(train_dataloader_callable=train_dataloader_callable, 
                               val_dataloader_callable=val_dataloader_callable, reader_pool_type=READER_POOL_TYPE, label_column="label", device_id=hvd.rank(), device_count=hvd.size(), batch_size=BATCH_SIZE, workers_count=WORKERS_COUNT)
  default_dir = f'/dbfs/Users/{username}/tmp/lightning'
  
  # `gpus` parameter here should be 1 because the parallelism is controlled by Horovod
  return train(hvd_model, hvd_datamodule, gpus=1, device_id=hvd.rank(), device_count=hvd.size(), batch_size=BATCH_SIZE, 
               mlflow_experiment_id=mlflow_experiment_id,
              default_dir=default_dir, train_steps_per_epoch=train_steps_per_epoch, val_steps_per_epoch=val_steps_per_epoch, workers_count=WORKERS_COUNT, 
      reader_pool_type=READER_POOL_TYPE, max_epochs=500)

# COMMAND ----------

# from sparkdl import HorovodRunner

# # with a negative np number, we will launch multi gpu training on one node
# hr = HorovodRunner(np=-1, driver_log_verbosity='all')
# hvd_model = hr.run(train_hvd)

# COMMAND ----------

from sparkdl import HorovodRunner

# This will launch a distributed training on np devices
hr = HorovodRunner(np=2, driver_log_verbosity='all')
hvd_model = hr.run(train_hvd)
