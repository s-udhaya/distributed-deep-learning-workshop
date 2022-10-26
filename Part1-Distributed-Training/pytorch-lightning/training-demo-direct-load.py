# Databricks notebook source
# MAGIC %run ../00_setup 

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# train_tbl_name = 'silvertrain'
# from delta.tables import DeltaTable
# deltaTable = DeltaTable.forName(spark, f'{database_name}.{train_tbl_name}')
# table_details = deltaTable.detail()
# display(table_details)
# train_rows = deltaTable.toDF().count()
# deltaTable.generate("symlink_format_manifest")
# from pathlib import Path
# table_location = table_details.select("location").collect()[0].location.replace("dbfs:", "/dbfs")
# Non spark enviroment
# train_parquet_files = Path(f"{table_location}/_symlink_format_manifest/manifest").read_text().splitlines()
# train_parquet_files = [parquet_file.replace("<external bucket path prefix>", "file:///dbfs") for parquet_file in train_parquet_files]
# in spark environment
train_table = spark.read.format("delta").table(f"{database_name}.silvertrain")
train_rows = train_table.count()
train_parquet_files = train_table.inputFiles()
train_parquet_files = [parquet_file.replace("dbfs:", "file:///dbfs") for parquet_file in train_parquet_files]
print(train_parquet_files)


# COMMAND ----------

# val_tbl_name = 'silverval'
# from delta.tables import DeltaTable
# deltaTable = DeltaTable.forName(spark, f'{database_name}.{val_tbl_name}')
# deltaTable.generate("symlink_format_manifest")
# Non spark enviroment
# from pathlib import Path
# table_location = table_details.select("location").collect()[0].location.replace("dbfs:", "/dbfs")
# Non spark enviroment
# val_parquet_files = Path(f"{table_location}/_symlink_format_manifest/manifest").read_text().splitlines()
# val_parquet_files = [parquet_file.replace("<external bucket path prefix>", "file:///dbfs") for parquet_file in val_parquet_files]
# in spark environment
val_table = spark.read.format("delta").table(f"{database_name}.silverval")
val_rows = val_table.count()
val_parquet_files = val_table.inputFiles()
val_parquet_files = [parquet_file.replace("dbfs:", "file:///dbfs") for parquet_file in val_parquet_files]
print(val_parquet_files)

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

from math import ceil
import mlflow
from functools import partial
from src.data_module import FlowersDataModule
from src.model import LitClassificationModel
from src.training import train
from src.mlflow_util import prepare_mlflow_experiment
from petastorm.pytorch import DataLoader, BatchedDataLoader
from petastorm import make_batch_reader

BATCH_SIZE = 32

READER_POOL_TYPE = "thread"
WORKERS_COUNT = 2


train_dataloader_callable = partial(data_loader, train_parquet_files)
val_dataloader_callable = partial(data_loader, val_parquet_files)

datamodule = FlowersDataModule(train_dataloader_callable=train_dataloader_callable, 
                               val_dataloader_callable=val_dataloader_callable, batch_size=BATCH_SIZE, workers_count=WORKERS_COUNT, reader_pool_type=READER_POOL_TYPE)

model = LitClassificationModel(class_count=5, learning_rate=1e-5, label_column="label_idx")

train_steps_per_epoch = ceil(train_rows //  BATCH_SIZE)
val_steps_per_epoch = ceil(val_rows //  BATCH_SIZE)

default_dir = f'/dbfs/Users/{username}/tmp/lightning'

prepare_mlflow_experiment(username, DATABRICKS_HOST, DATABRICKS_TOKEN, "pytorch-lightning-petastorm-direct-load-from-delta")

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
def train_hvd():
  hvd.init()
  
  # mlflow workaround to ensure that horovod subprocesses can find and connect to mlflow
  mlflow.set_tracking_uri("databricks")
  os.environ['DATABRICKS_HOST'] = DATABRICKS_HOST
  os.environ['DATABRICKS_TOKEN'] = DATABRICKS_TOKEN
  
  ## we pass the experiment id over to the workers
  mlflow_experiment_id = experiment.experiment_id

  batch_size = BATCH_SIZE * 2
  STEPS_PER_EPOCH = train_rows //  (batch_size *hvd.size())
  
  hvd_model = LitClassificationModel(class_count=5, learning_rate=1e-5*hvd.size(), 
                                     device_id=hvd.rank(), device_count=hvd.size())
  
  hvd_datamodule = FlowersDataModule(train_parquet_files, val_parquet_files, 
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
