# Databricks notebook source
# MAGIC %md
# MAGIC # 03. Model Training - Distributed 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In this notebook we will see how we can take our single node training code and scale training across our cluster.
# MAGIC 
# MAGIC We will:
# MAGIC - Use [Petastorm](https://docs.databricks.com/applications/machine-learning/load-data/ddl-data.html) to load our datasets from Delta and convert to tf.data datsets.
# MAGIC - Conduct single node training using Petastorm.
# MAGIC - Scale training across multiple nodes using [Horovod](https://docs.databricks.com/applications/machine-learning/train-model/distributed-training/horovod-runner.html) to orchestrate training, with Petastorm as a means to feed data to the training process.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Motivation

# COMMAND ----------

# MAGIC %md
# MAGIC A very large training problem may benefit from scaling out verticaly (training on a larger instance) or horizontally (using multiple machines). Scaling vertically may be an approach to adopt, but it can be costly to run a large instance for an extended period of time and scaling a single machine is limited (ie. there's only so many cores + RAM one machine could have). 
# MAGIC 
# MAGIC Scaling horizontally can be an affordable way to leverage the compute required to tackle very large training events. Multiple smaller instances may be more readibily available at cheaper rates in the cloud than a single, very large instance. Theoretically, you could add as many nodes as you wish (dependent on your cloud limits). In this section, we'll go over how to incorporate [Petastorm](https://docs.databricks.com/applications/machine-learning/load-data/ddl-data.html) and [Horovod](https://docs.databricks.com/applications/machine-learning/train-model/distributed-training/horovod-runner.html) into our single node training regime from to distribute training to multiple machines.

# COMMAND ----------

# MAGIC %md
# MAGIC Our first single node training example only used a fraction of the data and it required to work with the data sitting in memory. Typical datasets used for training may not fit in memory for a single machine. Petastorm enables directly loading data stored in parquet format, meaning we can go from our silver Delta table to a distributed `torch.util.data.Dataset` without having to copy our table into a `Pandas` dataframe and wasting additional memory.
# MAGIC 
# MAGIC <a href="https://github.com/uber/petastorm" target="_blank">Petastorm</a> enables single machine or distributed training and evaluation of deep learning models from datasets in Apache Parquet format and datasets that are already loaded as Spark DataFrames. It supports ML frameworks such as TensorFlow, Pytorch, and PySpark and can be used from pure Python code.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Here we use `petastorm` to load and cache data that was read directly from our silver train and val Delta tables:

# COMMAND ----------

# MAGIC %run ../../Part0-setup-and-data-preparation/00_setup

# COMMAND ----------

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

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK

import horovod.torch as hvd
from sparkdl import HorovodRunner

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## i. Configs

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Set global variables

# COMMAND ----------

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

BATCH_SIZE = 32
EPOCHS = 3

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## ii. Load prepared data from Silver layer 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Load the train and validation datasets from the Silver layer we created in [01_data_prep]($./01_data_prep) as Spark DataFrames.

# COMMAND ----------

cols_to_keep = ['content', 'label_idx']

train_tbl_name = 'silver_train'
val_tbl_name = 'silver_val'

train_df = (spark.table(f'{database_name}.{train_tbl_name}')
            .select(cols_to_keep))

val_df = (spark.table(f'{database_name}.{val_tbl_name}')
          .select(cols_to_keep))

# Make sure the number of partitions is at least the number of workers which is required for distributed training.
train_df = train_df.repartition(2)
val_df = val_df.repartition(2)

print('train_df count:', train_df.count())
print('val_df count:', val_df.count())

# COMMAND ----------

num_classes = train_df.select('label_idx').distinct().count()

print('Number of classes:', num_classes)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## iii. Convert the Spark DataFrame to a pytorch Dataset
# MAGIC In order to convert Spark DataFrames to a pytorch datasets, we need to do it in two steps:
# MAGIC <br><br>
# MAGIC 
# MAGIC 
# MAGIC 0. Define where you want to copy the data by setting Spark config
# MAGIC 0. Call `make_spark_converter()` method to make the conversion
# MAGIC 
# MAGIC This will copy the data to the specified path.

# COMMAND ----------

train_df.display()

# COMMAND ----------

dbutils.fs.rm(f'/tmp/distributed_dl_workshop_{user}/petastorm', recurse=True)
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, f'file:///dbfs/tmp/distributed_dl_workshop_{user}/petastorm')

train_petastorm_converter = make_spark_converter(train_df)
val_petastorm_converter = make_spark_converter(val_df)

train_size = len(train_petastorm_converter)
val_size = len(val_petastorm_converter)

# COMMAND ----------

# MAGIC %md
# MAGIC ## iv. Define model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We will define the same model as in [`02_model_training_single_node`]($./02_model_training_single_node), along with the same preprocessing functionality.

# COMMAND ----------

def build_model(num_classes):
    from torchvision import models
    import torch.nn as nn
    model = models.mobilenet_v2(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model

# COMMAND ----------

def transform_row(is_train, pd_batch):
  """
  The input and output of this function must be pandas dataframes.
  Do data augmentation for the training dataset only.
  """
  transformers = [transforms.Lambda(lambda x: Image.open(io.BytesIO(x)))]
  if is_train:
    transformers.extend([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
    ])
  else:
    transformers.extend([
      transforms.Resize(256),
      transforms.CenterCrop(224),
    ])
  transformers.extend([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])
  
  trans = transforms.Compose(transformers)
  
  pd_batch['features'] = pd_batch['content'].map(lambda x: trans(x).numpy())
  pd_batch = pd_batch.drop(labels=['content'], axis=1)
  return pd_batch

def get_transform_spec(is_train=True):
  # Note that the output shape of the `TransformSpec` is not automatically known by petastorm, 
  # so we need to specify the shape for new columns in `edit_fields` and specify the order of 
  # the output columns in `selected_fields`.
  return TransformSpec(partial(transform_row, is_train), 
                       edit_fields=[('features', np.float32, (3, 224, 224), False)], 
                       selected_fields=['features', 'label_idx'])

# COMMAND ----------

def train_one_epoch(model, criterion, optimizer, scheduler, 
                    train_dataloader_iter, steps_per_epoch, epoch, 
                    device):
  model.train()  # Set model to training mode

  # statistics
  running_loss = 0.0
  running_corrects = 0

  # Iterate over the data for one epoch.
  for step in range(steps_per_epoch):
    pd_batch = next(train_dataloader_iter)
    inputs, labels = pd_batch['features'].to(device), pd_batch['label_idx'].type(torch.LongTensor).to(device)

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

  epoch_loss = running_loss / (steps_per_epoch * BATCH_SIZE)
  epoch_acc = running_corrects.double() / (steps_per_epoch * BATCH_SIZE)

  print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
  return epoch_loss, epoch_acc

def evaluate(model, criterion, val_dataloader_iter, validation_steps, device, 
             metric_agg_fn=None):
  model.eval()  # Set model to evaluate mode

  # statistics
  running_loss = 0.0
  running_corrects = 0

  # Iterate over all the validation data.
  for step in range(validation_steps):
    pd_batch = next(val_dataloader_iter)
    inputs, labels = pd_batch['features'].to(device), pd_batch['label_idx'].type(torch.LongTensor).to(device)
    # Do not track history in evaluation to save memory
    with torch.set_grad_enabled(False):
      # forward
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)

    # statistics
    running_loss += loss.item()
    running_corrects += torch.sum(preds == labels.data)
  
  # The losses are averaged across observations for each minibatch.
  epoch_loss = running_loss / validation_steps
  epoch_acc = running_corrects.double() / (validation_steps * BATCH_SIZE)
  
  # metric_agg_fn is used in the distributed training to aggregate the metrics on all workers
  if metric_agg_fn is not None:
    epoch_loss = metric_agg_fn(epoch_loss, 'avg_loss')
    epoch_acc = metric_agg_fn(epoch_acc, 'avg_acc')

  print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
  return epoch_loss, epoch_acc

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## v. Single Node - Feed data to single node using Petastorm
# MAGIC 
# MAGIC We will first do single node training where we will use Petastorm to feed the data from Delta to the training process. We need to use Petastorm's <a href="https://petastorm.readthedocs.io/en/latest/api.html#petastorm.spark.spark_dataset_converter.SparkDatasetConverter.make_tf_dataset" target="_blank">make_tf_dataset</a> to read batches of data.<br><br>
# MAGIC 
# MAGIC * Note that we use **`num_epochs=None`** to generate infinite batches of data to avoid handling the last incomplete batch. This is particularly useful in the distributed training scenario, where we need to guarantee that the numbers of data records seen on all workers are identical. Given that the length of each data shard may not be identical, setting **`num_epochs`** to any specific number would fail to meet the guarantee.
# MAGIC * The **`workers_count`** param specifies the number of threads or processes to be spawned in the reader pool, and it is not a Spark worker. 

# COMMAND ----------

def train_and_evaluate(lr=0.001):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  model = build_model(num_classes=num_classes)
  model = model.to(device)

  criterion = torch.nn.CrossEntropyLoss()

  # Only parameters of final layer are being optimized.
  optimizer = torch.optim.SGD(model.classifier[1].parameters(), lr=lr, momentum=0.9)

  # Decay LR by a factor of 0.1 every 7 epochs
  exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
  
  with train_petastorm_converter.make_torch_dataloader(transform_spec=get_transform_spec(is_train=True), 
                                             batch_size=BATCH_SIZE) as train_dataloader, \
       val_petastorm_converter.make_torch_dataloader(transform_spec=get_transform_spec(is_train=False), 
                                           batch_size=BATCH_SIZE) as val_dataloader:
    
    train_dataloader_iter = iter(train_dataloader)
    steps_per_epoch = len(train_petastorm_converter) // BATCH_SIZE
    
    val_dataloader_iter = iter(val_dataloader)
    validation_steps = max(1, len(val_petastorm_converter) // BATCH_SIZE)
    
    for epoch in range(EPOCHS):
      print('Epoch {}/{}'.format(epoch + 1, EPOCHS))
      print('-' * 10)

      train_loss, train_acc = train_one_epoch(model, criterion, optimizer, exp_lr_scheduler, 
                                              train_dataloader_iter, steps_per_epoch, epoch, 
                                              device)
      val_loss, val_acc = evaluate(model, criterion, val_dataloader_iter, validation_steps, device)

  return val_loss
  
loss = train_and_evaluate()

# COMMAND ----------

# MAGIC %md
# MAGIC You can see the code to train our model is exactly the same as our single node training. Adding Petastorm to load our data directly from the underlying parquet files in our delta tables simply required using the `make_spark_converter` class and initializing our datasets using the `make_torch_dataloader()` method.
# MAGIC 
# MAGIC Next, we need to distribute the training across our cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ## v. Distribute training with Horovod
# MAGIC 
# MAGIC [Horovod](https://github.com/horovod/horovod) is a distributed training framework for TensorFlow, Keras, and PyTorch. Databricks supports distributed deep learning training using HorovodRunner and the `horovod.spark` package which we will use next. 
# MAGIC 
# MAGIC We can use Horovod to train across multiple machines, meaning we can distribute training across CPU-clusters or GPU clusters. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC [HorovodRunner](https://databricks.github.io/spark-deep-learning/#sparkdl.HorovodRunner) is a general API to run distributed DL workloads on Databricks using Uber’s <a href="https://github.com/uber/horovod" target="_blank">Horovod</a> framework. By integrating Horovod with Spark’s barrier mode, Databricks is able to provide higher stability for long-running deep learning training jobs on Spark. 
# MAGIC 
# MAGIC ### How it works
# MAGIC * HorovodRunner takes a Python method that contains DL training code with Horovod hooks. 
# MAGIC * This method gets pickled on the driver and sent to Spark workers. 
# MAGIC * A Horovod MPI job is embedded as a Spark job using barrier execution mode. 
# MAGIC * The first executor collects the IP addresses of all task executors using BarrierTaskContext and triggers a Horovod job using mpirun. 
# MAGIC * Each Python MPI process loads the pickled program back, deserializes it, and runs it.
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/horovod-runner.png)
# MAGIC 
# MAGIC For additional resources, see:
# MAGIC * <a href="https://docs.microsoft.com/en-us/azure/databricks/applications/deep-learning/distributed-training/horovod-runner" target="_blank">Horovod Runner Docs</a>
# MAGIC * <a href="https://vimeo.com/316872704/e79235f62c" target="_blank">Horovod Runner webinar</a>  

# COMMAND ----------

# MAGIC %md
# MAGIC Inside our function, we'll incorporate the same logic we used in the single node and petastorm training examples plus additional logic to account for GPUs. 
# MAGIC 
# MAGIC If you have a CPU cluster, then these lines will be ignored and training will occur on CPU.

# COMMAND ----------

import mlflow
import os

BATCH_SIZE = 8
def metric_average(val, name):
  tensor = torch.tensor(val)
  avg_tensor = hvd.allreduce(tensor, name=name)
  return avg_tensor.item()

def train_and_evaluate_hvd(lr=0.001):
  hvd.init()  # Initialize Horovod.
  
  # To enable tracking from the Spark workers to Databricks
  mlflow.mlflow.set_tracking_uri('databricks')
  os.environ['DATABRICKS_HOST'] = DATABRICKS_HOST
  os.environ['DATABRICKS_TOKEN'] = DATABRICKS_TOKEN
    
  # Horovod: pin GPU to local rank.
  if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())
    device = torch.cuda.current_device()
  else:
    device = torch.device("cpu")
  
  model = build_model(num_classes=num_classes)
  model = model.to(device)

  criterion = torch.nn.CrossEntropyLoss()
  
  # Effective batch size in synchronous distributed training is scaled by the number of workers.
  # An increase in learning rate compensates for the increased batch size.
  optimizer = torch.optim.SGD(model.classifier[1].parameters(), lr=lr * hvd.size(), momentum=0.9)
  
  # Broadcast initial parameters so all workers start with the same parameters.
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)
  hvd.broadcast_optimizer_state(optimizer, root_rank=0)
  
  # Wrap the optimizer with Horovod's DistributedOptimizer.
  optimizer_hvd = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

  exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_hvd, step_size=7, gamma=0.1)

  with train_petastorm_converter.make_torch_dataloader(transform_spec=get_transform_spec(is_train=True), 
                                             cur_shard=hvd.rank(), shard_count=hvd.size(),
                                             batch_size=BATCH_SIZE) as train_dataloader, \
       val_petastorm_converter.make_torch_dataloader(transform_spec=get_transform_spec(is_train=False),
                                           cur_shard=hvd.rank(), shard_count=hvd.size(),
                                           batch_size=BATCH_SIZE) as val_dataloader:
    
    train_dataloader_iter = iter(train_dataloader)
    steps_per_epoch = len(train_petastorm_converter) // (BATCH_SIZE * hvd.size())
    
    val_dataloader_iter = iter(val_dataloader)
    validation_steps = max(1, len(val_petastorm_converter) // (BATCH_SIZE * hvd.size()))
    
    for epoch in range(EPOCHS):
      print('Epoch {}/{}'.format(epoch + 1, EPOCHS))
      print('-' * 10)

      train_loss, train_acc = train_one_epoch(model, criterion, optimizer_hvd, exp_lr_scheduler, 
                                              train_dataloader_iter, steps_per_epoch, epoch, 
                                              device)
      val_loss, val_acc = evaluate(model, criterion, val_dataloader_iter, validation_steps,
                                   device, metric_agg_fn=metric_average)
    
            # MLflow Tracking (Log only from Worker 0)
    if hvd.rank() == 0:
        # Log events to MLflow
        with mlflow.start_run(run_id = active_run_uuid):
            # Log MLflow Parameters
            mlflow.log_param('epochs', EPOCHS)
            mlflow.log_param('batch_size', BATCH_SIZE)

            # Log MLflow Metrics
            mlflow.log_metric('val_loss', val_loss)
            mlflow.log_metric('val_accuracy', val_acc)

            # Log Model
            mlflow.pytorch.log_model(model, 'model')

  return val_loss

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Training on the driver with Horovod

# COMMAND ----------

# MAGIC %md Test it out on just the driver.
# MAGIC 
# MAGIC `np=-1` will force Horovod to run on a single core on the Driver node.

# COMMAND ----------


with mlflow.start_run(run_name='horovod_driver') as run:
  active_run_uuid = mlflow.active_run().info.run_uuid
  hr = HorovodRunner(np=-1)  
  hr.run(train_and_evaluate_hvd)
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Distributed Training with Horovod

# COMMAND ----------

## OPTIONAL: You can enable Horovod Timeline as follows, but can incur slow down from frequent writes, and have to export out of Databricks to upload to chrome://tracing
# import os
# os.environ["HOROVOD_TIMELINE"] = f"{working_dir}/_timeline.json"
with mlflow.start_run(run_name='horovod_distributed') as run:
  active_run_uuid = mlflow.active_run().info.run_uuid
  hr = HorovodRunner(np=2)   # We assume cluster consists of two workers.
  hr.run(train_and_evaluate_hvd)
mlflow.end_run()

# COMMAND ----------

# MAGIC %md Finally, we <a href="https://petastorm.readthedocs.io/en/latest/api.html#petastorm.spark.spark_dataset_converter.SparkDatasetConverter.delete" target="_blank">delete</a> the cached Petastorm files.

# COMMAND ----------

train_petastorm_converter.delete()
val_petastorm_converter.delete()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## vi. Load model
# MAGIC 
# MAGIC Load model from MLflow

# COMMAND ----------

trained_model = mlflow.pytorch.load_model(f'runs:/{run.info.run_id}/model')
trained_model
