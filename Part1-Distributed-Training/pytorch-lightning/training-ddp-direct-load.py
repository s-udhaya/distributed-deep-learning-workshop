# Databricks notebook source
# MAGIC %run ../../Part0-setup-and-data-preparation/00_setup

# COMMAND ----------

# MAGIC %run ./source_modules

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

NUM_DEVICES = 2
GPU_PER_DEVICE = 4
WORLD_SIZE = NUM_DEVICES #* GPU_PER_DEVICE
BATCH_SIZE = 4

READER_POOL_TYPE = "dummy"
WORKERS_COUNT = 1
MAX_EPOCHS = 500

# COMMAND ----------

train_table = spark.read.format("delta").table(f"{database_name}.silvertrain")
train_rows = train_table.count()
train_parquet_files = train_table.inputFiles()
train_parquet_files = [parquet_file.replace("dbfs:", "file:///dbfs") for parquet_file in train_parquet_files]

# COMMAND ----------

val_table = spark.read.format("delta").table(f"{database_name}.silverval")
val_rows = val_table.count()
val_parquet_files = val_table.inputFiles()
val_parquet_files = [parquet_file.replace("dbfs:", "file:///dbfs") for parquet_file in val_parquet_files]

# COMMAND ----------

from petastorm.pytorch import DataLoader, BatchedDataLoader
from petastorm import make_batch_reader
from petastorm.spark.spark_dataset_converter import TorchDatasetContextManager

def data_loader(parquet_files, transform_spec,
                num_epochs,
                workers_count,
                cur_shard,
                shard_count,
              reader_pool_type,
                batch_size):
  
  petastorm_reader_kwargs = {"num_epochs": num_epochs,
                                                   "transform_spec": transform_spec,
                                                   "cur_shard": cur_shard, 
                                                   "shard_count": shard_count, 
                                                   "workers_count": workers_count, 
                                                   "reader_pool_type": reader_pool_type}

  dataloader = TorchDatasetContextManager(parquet_files, batch_size, petastorm_reader_kwargs, 0, None)

#   dataloader = BatchedDataLoader(make_batch_reader(parquet_files, num_epochs=num_epochs,
#                                                    transform_spec=transform_spec,
#                                                    cur_shard=cur_shard, 
#                                                    shard_count=shard_count, 
#                                                    workers_count=workers_count, 
#                                                    reader_pool_type=reader_pool_type), 
#                                  batch_size)
  return dataloader

# COMMAND ----------

import os
import mlflow
from math import ceil
from spark_pytorch_distributor import MirrorRunner
from functools import partial


def train_mirrored():
    ## we pass the experiment id over to the workers
  train_dataloader_callable = partial(data_loader, train_parquet_files)
  val_dataloader_callable = partial(data_loader, val_parquet_files)

  # mlflow workaround to ensure that horovod subprocesses can find and connect to mlflow
  mlflow.set_tracking_uri("databricks")
  experiment = prepare_mlflow_experiment(username, DATABRICKS_HOST, DATABRICKS_TOKEN, "pytorch-lightning-petastorm-dataframe-load")
  print(f"node rank{int(os.environ['NODE_RANK'])}")
  ## we pass the experiment id over to the workers
  mlflow_experiment_id = experiment.experiment_id
  mlflow_experiment_id = experiment.experiment_id

  train_steps_per_epoch = ceil(train_rows //  (BATCH_SIZE * WORLD_SIZE))
  val_steps_per_epoch = ceil(val_rows //  (BATCH_SIZE * WORLD_SIZE)) 
  
 
  model = LitClassificationModel(class_count=5, learning_rate=1e-5*WORLD_SIZE, 
                                       device_id=int(os.environ["NODE_RANK"]), device_count=WORLD_SIZE, label_column="label_idx")
  datamodule = FlowersDataModule(train_dataloader_callable=train_dataloader_callable, 
                                 val_dataloader_callable=val_dataloader_callable, reader_pool_type=READER_POOL_TYPE, device_id=int(os.environ["NODE_RANK"]), device_count=WORLD_SIZE, batch_size=BATCH_SIZE, workers_count=WORKERS_COUNT)
  default_dir = f'/dbfs/Users/{username}/tmp/lightning'

    # `gpus` parameter here should be 1 because the parallelism is controlled by Horovod
  return train(model, datamodule, gpus=1, num_nodes=2, strategy="ddp", device_id=int(os.environ["NODE_RANK"]), device_count=8, batch_size=BATCH_SIZE, 
                 mlflow_experiment_id=mlflow_experiment_id,
                default_dir=default_dir, train_steps_per_epoch=train_steps_per_epoch, val_steps_per_epoch=val_steps_per_epoch, workers_count=WORKERS_COUNT, 
        reader_pool_type=READER_POOL_TYPE, max_epochs=10)

# COMMAND ----------

model = MirrorRunner(num_slots=2, use_custom_strategy=True, local_mode=False).run(train_mirrored)
