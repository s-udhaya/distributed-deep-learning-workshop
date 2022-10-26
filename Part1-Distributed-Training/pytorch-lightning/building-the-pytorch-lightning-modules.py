# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Building the PyTorch Lightning Model
# MAGIC 
# MAGIC **[NOTE]**
# MAGIC 
# MAGIC Separate callback classes can be used for sidecar operations like logging, etc but we decided to keep evething within the model class for this example.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Model Definition
# MAGIC 
# MAGIC 
# MAGIC PyTorch Lightning model code can be organised into the Lightning Module class
# MAGIC this gives a convenient wrapper for the Model definition and the optimisers
# MAGIC 
# MAGIC On DBR 10.4 ML LTS set the version to 1.6.5 ie
# MAGIC `%pip install pytorch_lightning==1.6.5 torchmetrics`
# MAGIC 
# MAGIC On DBR 11.1 ML set the version to 1.7.2 ie
# MAGIC `%pip install pytorch_lightning==1.7.2 torchmetrics`

# COMMAND ----------

# MAGIC %pip install pytorch_lightning==1.6.5 torchmetrics

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
              device_id:int=0, device_count:int=1, family:str='mobilenet'):
    
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

# MAGIC %md
# MAGIC 
# MAGIC ## DataLoader Class
# MAGIC 
# MAGIC This class holds all the logic for processing and loading the dataset.
# MAGIC 
# MAGIC **[NOTE]**
# MAGIC 
# MAGIC The value of parameter `num_epochs` used in `make_torch_dataloader` function is deliberately set it to `None` (it is also a default value) in order to generate an infinite number of data batches to avoid handling the last, likely incomplete, batch. This is especially important for distributed training where we need to guarantee that the numbers of data records seen on all workers are identical per step. Given that the length of each data shard may not be identical, setting `num_epochs` to any specific number would fail to meet the guarantee and can result in an error. Even though this may not be really important for training on a single device, it determines the way we control epochs (training will run forever on infinite dataset which means there would be only 1 epoch if other means of controlling the epoch duration are not used), so we decided to introduce it here from the beginning.
# MAGIC 
# MAGIC Setting the value of `num_epochs=None` is also important for the validation process. At the time this notebook was developed, Pytorch Lightning Trainer will run a sanity validation check prior to any training, unless instructed otherwise (i.e. `num_sanity_val_steps` is set to `0`). That sanity check will initialise the validation data loader and will read the `num_sanity_val_steps` batches from it before the first training epoch. Training will not reload the validation dataset for the actual validation phase of the first epoch which will result in error (an attempt to read a second time from data loader which was not completed in the previous attempt). Possible workarounds to avoid this issue is using a finite amount of epochs in `num_epochs` (e.g. `num_epochs=1` as there is no point in evaluating on repeated dataset), which is not ideal as it will likely result in a last batch being smaller than other batches and at the time when this notebook was developed there was no way of setting an equivalent of `drop_last` for the Data Loader created by `make_torch_dataloader`. The only way we found to work around this was to avoid doing any sanity checks (i.e. setting `num_sanity_val_steps=0`, setting it to anything else doesn't work) and using `limit_val_batches` parameter of the Trainer class to avoid the infinitely running validation.

# COMMAND ----------

from petastorm import TransformSpec
from PIL import Image
from torchvision import transforms
import numpy as np
import io

from pyspark.sql.functions import col, pandas_udf, PandasUDFType


class FlowersDataModule(pl.LightningDataModule):
  
  def __init__(self, train_converter, val_converter, device_id:int=0, device_count:int=1):
    
    self.train_converter = train_converter
    self.val_converter = val_converter
    self.train_dataloader_context = None
    self.val_dataloader_context = None
    self.prepare_data_per_node = False
    self._log_hyperparams = False
    
    self.device_id = device_id
    self.device_count = device_count
  
    
  def train_dataloader(self):
    if self.train_dataloader_context:
        self.train_dataloader_context.__exit__(None, None, None)
    self.train_dataloader_context = self.train_converter.make_torch_dataloader(transform_spec=self._get_transform_spec(), 
                                                                               num_epochs=None,
                                                                               cur_shard=self.device_id, 
                                                                               shard_count=self.device_count, 
                                                                               batch_size=BATCH_SIZE*self.device_count)
    return self.train_dataloader_context.__enter__()
  
  def val_dataloader(self):
    if self.val_dataloader_context:
        self.val_dataloader_context.__exit__(None, None, None)
    self.val_dataloader_context = self.val_converter.make_torch_dataloader(transform_spec=self._get_transform_spec(), 
                                                                           num_epochs=None, 
                                                                           cur_shard=self.device_id, 
                                                                           shard_count=self.device_count,
                                                                           batch_size=BATCH_SIZE*self.device_count)
    return self.val_dataloader_context.__enter__()
    
  def teardown(self, stage=None):
    # Close all readers (especially important for distributed training to prevent errors)
    self.train_dataloader_context.__exit__(None, None, None)
    self.val_dataloader_context.__exit__(None, None, None)
    
  
  def preprocess(self, img):
    
    image = Image.open(io.BytesIO(img))
    transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transform(image)
  
  
  def _transform_rows(self, batch):
    
    # To keep things simple, use the same transformation both for training and validation
    batch["features"] = batch["content"].map(lambda x: self.preprocess(x).numpy())
    batch = batch.drop(labels=["content"], axis=1)
    return batch

  def _get_transform_spec(self):
    return TransformSpec(self._transform_rows, 
                         edit_fields=[("features", np.float32, (3, 224, 224), False)], 
                         selected_fields=["features", "label"])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Main Training Loop

# COMMAND ----------

import datetime as dt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
import mlflow


def report_duration(action, start):
  """
  
  Helper function in order to assist in benchmarking the code.
  
  """
  
  end = dt.datetime.now()
  ds = (end - start).total_seconds()
  h, rem = divmod(ds, 3600)
  m, s = divmod(rem, 60)
  if h > 0:
    run_time = "{} hours {} minutes".format(int(h), int(m))
  elif m > 0:
    run_time = "{} minutes {} seconds".format(int(m), int(s))
  else:
    run_time = "{} seconds".format(int(s))

  msg = f"{action} completed in ***{run_time}***"
  print(msg)


def train(model, dataloader, gpus:int=0, 
          strategy:str=None, device_id:int=0, 
          device_count:int=1, logging_level=logging.INFO,
          default_dir:str='/dbfs/tmp/trainer_logs',
          ckpt_restore:str=None,
          mlflow_experiment_id:str=None):
  
  start = dt.datetime.now()

  if device_id == 0:
    
    # we trigger autolog here to ensure we capture all the params and the training process
    mlflow.pytorch.autolog()
    
    device = str(max(gpus, device_count)) + ' GPU' + ('s' if gpus > 1 or device_count > 1 else '') if gpus > 0  else 'CPU'
    print(f"Train on {device}:")
    print(f"- max epoch count: {MAX_EPOCH_COUNT}")
    print(f"- batch size: {BATCH_SIZE*device_count}")
    #print(f"- steps per epoch: {STEPS_PER_EPOCH}")
    print("\n======================\n")
  
  # Use check_on_train_epoch_end=True to evaluate at the end of each epoch
  verbose = True if device_id == 0 else False
  stopper = EarlyStopping(monitor="val_loss", min_delta=EARLY_STOP_MIN_DELTA, patience=EARLY_STOP_PATIENCE,
                          stopping_threshold=0.55,
                          verbose=verbose, mode='min', check_on_train_epoch_end=True)
  callbacks = [stopper]
  
  
  # You could also use an additinal progress bar but default progress reporting was sufficient. Uncomment next line if desired
  # callbacks.append(TQDMProgressBar(refresh_rate=STEPS_PER_EPOCH, process_position=0))
  
  # We could use `on_train_batch_start` to control epoch sizes as shown in the link below but it's cleaner when 
  # done here with `limit_train_batches` parameter
  # https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/core/hooks.html#ModelHooks.on_train_batch_start
  trainer = pl.Trainer(
      gpus=gpus,
      max_epochs=MAX_EPOCH_COUNT,
      limit_train_batches=STEPS_PER_EPOCH,  # this is the way to end the epoch
      log_every_n_steps=1,
      val_check_interval=STEPS_PER_EPOCH,  # this value must be the same as `limit_train_batches`
      num_sanity_val_steps=0,  # this must be zero to prevent a Petastorm error about Data Loader not being read completely
      limit_val_batches=1,  # any value would work here but there is point in validating on repeated set of data
      reload_dataloaders_every_n_epochs=1,  # need to set this to 1
      strategy=strategy,
      callbacks=callbacks,
      default_root_dir=default_dir  
  )
  
  if device_id == 0:
    with mlflow.start_run(experiment_id=mlflow_experiment_id) as run:
      trainer.fit(model, dataloader, ckpt_path=ckpt_restore)
      report_duration(f"Training", start)
      print("\n\n---------------------")
  else:
    trainer.fit(model, dataloader, ckpt_path=ckpt_restore)
      
  
  return model.model if device_id == 0 else None

# COMMAND ----------


