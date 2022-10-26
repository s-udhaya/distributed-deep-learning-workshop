# Databricks notebook source
# MAGIC %run ../00_setup 

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

from petastorm import TransformSpec
from PIL import Image
from torchvision import transforms
import numpy as np
import io

from pyspark.sql.functions import col, pandas_udf, PandasUDFType


class FlowersDataModule(pl.LightningDataModule):

    def __init__(self, train_converter, val_converter, device_id: int = 0, device_count: int = 1, batch_size: int = 16,
                 workers_count: int = 1):

        self.train_converter = train_converter
        self.val_converter = val_converter
        self.train_dataloader_context = None
        self.val_dataloader_context = None
        self.prepare_data_per_node = False
        self._log_hyperparams = False

        self.device_id = device_id
        self.device_count = device_count
        self.batch_size = batch_size
        self.workers_count = workers_count

    def train_dataloader(self):
        if self.train_dataloader_context:
            self.train_dataloader_context.__exit__(None, None, None)
        else:
            self.train_dataloader_context = self.train_converter.make_torch_dataloader(
                transform_spec=self._get_transform_spec(),
                num_epochs=None,
                workers_count=self.workers_count,
                cur_shard=self.device_id,
                shard_count=self.device_count,
              reader_pool_type="dummy",
                batch_size=self.batch_size)
        return self.train_dataloader_context.__enter__()

    def val_dataloader(self):
        if self.val_dataloader_context:
            self.val_dataloader_context.__exit__(None, None, None)
        else:
            self.val_dataloader_context = self.val_converter.make_torch_dataloader(
                transform_spec=self._get_transform_spec(),
                num_epochs=None,
                workers_count=self.workers_count,
                cur_shard=self.device_id,
                shard_count=self.device_count,
              reader_pool_type="dummy",
                batch_size=self.batch_size)
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
          device_count:int=1, batch_size:int=16, steps_per_epoch:int=1, max_epochs:int=1000, logging_level=logging.INFO,
          default_dir:str='/dbfs/tmp/trainer_logs',
          ckpt_restore:str=None,
          mlflow_experiment_id:str=None):
  
  start = dt.datetime.now()

  if device_id == 0:
    
    # we trigger autolog here to ensure we capture all the params and the training process
    mlflow.pytorch.autolog()
    
    device = str(max(gpus, device_count)) + ' GPU' + ('s' if gpus > 1 or device_count > 1 else '') if gpus > 0  else 'CPU'
    print(f"Train on {device}:")
    print(f"- max epoch count: {max_epochs}")
    print(f"- batch size: {batch_size}")
    print(f"- steps per epoch: {steps_per_epoch}")
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
      max_epochs=max_epochs,
      limit_train_batches=steps_per_epoch,  # this is the way to end the epoch
      log_every_n_steps=1,
      val_check_interval=steps_per_epoch,  # this value must be the same as `limit_train_batches`
      num_sanity_val_steps=0,  # this must be zero to prevent a Petastorm error about Data Loader not being read completely
      limit_val_batches=1,  # any value would work here but there is point in validating on repeated set of data
      reload_dataloaders_every_n_epochs=1,  # need to set this to 1
      strategy=strategy,
      callbacks=callbacks,
      default_root_dir=default_dir,
      enable_progress_bar=False,
      enable_model_summary=False,
      enable_checkpointing=False,
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

MAX_EPOCH_COUNT = 10
BATCH_SIZE = 32
STEPS_PER_EPOCH = 15

# When using databricks repos, it is not possible to write into working directories
# specifying a dbfs default dir helps to avoid this
default_dir = f'/dbfs/Users/{username}/tmp/lightning'
experiment_path = f'/Users/{username}/pytorch-lightning-on-databricks'

EARLY_STOP_MIN_DELTA = 0.01
EARLY_STOP_PATIENCE = 10

NUM_DEVICES = 2


import os
import mlflow

## Specifying the mlflow host server and access token 
# We put them to a variable to feed into horovod later on
db_host = "https://e2-demo-tokyo.cloud.databricks.com/"  # CHANGE THIS!
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# We put them into these environment variables as this is where mlflow will look by default
os.environ['DATABRICKS_HOST'] = db_host
os.environ['DATABRICKS_TOKEN'] = db_token

# We manually create the experiment so that we know the id and can send that to the worker nodes when we scale
experiment = mlflow.set_experiment(experiment_path)



# COMMAND ----------

from petastorm.spark import SparkDatasetConverter, make_spark_converter
import pyspark.sql.types as T
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
import pyspark.sql.types as T
from pyspark.sql.functions import udf

def to_class_index(classes: list, class_name:str):
    
    """
    Converts classes to a class_index so that we can create a tensor object
    """
    return classes.index(class_name)
  
  
def udf_class(class_list: list): 
    """
    Results has to be a longtype
    """
    return udf(lambda x: to_class_index(class_list, x), T.LongType())
    
    
def prepare_data(data_dir: str, num_devices: int):
    """
    This function loads the dataset and converts the label into a numeric index
    
    PyTorch Lightning suggests splitting datasets in the setup command but with petastorm we only need to do this once.
    Also spark is already distributed.
    
    """
    
    flowers_dataset = spark.read.format("delta").load(data_dir).select('content', 'label')
    
    
    flowers_dataset = flowers_dataset.repartition(num_devices*2)
    classes = list(flowers_dataset.select("label").distinct().toPandas()["label"])
    print(f'Num Classes: {len(classes)}')
    
    #class_index = udf(_to_class_index, T.LongType())  
    flowers_dataset = flowers_dataset.withColumn("label", udf_class(classes)(col("label")) )
    
    #### the following code is to make sure to sample each class by the proportion that it appears in the original dataset
    # Spark doesn't include a Stratified Sampling class hence we have to calculate what fraction each class is.
    total_size = flowers_dataset.count()
    print(f"Dataset size: {total_size}")
    
    groups = flowers_dataset.groupBy('label').count()
    groups = groups.withColumn('fraction', col('count')/total_size)
    
    fractions = groups.select('label', 'fraction').toPandas()
    fractions.set_index('label')
    
    val_df = flowers_dataset.sampleBy("label", fractions=fractions.fraction.to_dict(), seed=12)
    train_df = flowers_dataset.join(val_df, flowers_dataset.content==val_df.content, "leftanti")
    print(f"Train Size = {train_df.count()} Val Size = {val_df.count()}")

    train_converter = make_spark_converter(train_df)
    val_converter = make_spark_converter(val_df)
    
    return flowers_dataset, train_converter, val_converter 

# COMMAND ----------

dbutils.fs.rm("file:///dbfs/tmp/petastorm/cache", True)

Data_Directory = '/databricks-datasets/flowers/delta'

CACHE_DIR = "file:///dbfs/tmp/petastorm/cache"
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, CACHE_DIR)

flowers_df, train_converter, val_converter = prepare_data(data_dir=Data_Directory, 
                                                          num_devices=NUM_DEVICES)



# COMMAND ----------

BATCH_SIZE = 32
datamodule = FlowersDataModule(train_converter=train_converter, 
                               val_converter=val_converter, batch_size=BATCH_SIZE, workers_count=1)
model = LitClassificationModel(class_count=5, learning_rate=1e-5)
STEPS_PER_EPOCH = len(train_converter) //  BATCH_SIZE
train(model, datamodule, gpus=1, default_dir=default_dir, batch_size=BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH, max_epochs=1000)

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
def train_hvd():
  hvd.init()
  
  # mlflow workaround to ensure that horovod subprocesses can find and connect to mlflow
  mlflow.set_tracking_uri("databricks")
  os.environ['DATABRICKS_HOST'] = DATABRICKS_HOST
  os.environ['DATABRICKS_TOKEN'] = DATABRICKS_TOKEN
  
  ## we pass the experiment id over to the workers
  mlflow_experiment_id = experiment.experiment_id
  batch_size = BATCH_SIZE * 4
  STEPS_PER_EPOCH = len(train_converter) //  (batch_size *hvd.size())
  
  hvd_model = LitClassificationModel(class_count=5, learning_rate=1e-5*hvd.size(), 
                                     device_id=hvd.rank(), device_count=hvd.size())
  
  hvd_datamodule = FlowersDataModule(train_converter, val_converter, 
                                     device_id=hvd.rank(), device_count=hvd.size(), batch_size=batch_size, workers_count=2)
  
  # `gpus` parameter here should be 1 because the parallelism is controlled by Horovod
  return train(hvd_model, hvd_datamodule, gpus=1, device_id=hvd.rank(), device_count=hvd.size(), batch_size=batch_size, steps_per_epoch=STEPS_PER_EPOCH, 
               mlflow_experiment_id=mlflow_experiment_id,
              default_dir=default_dir)

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
