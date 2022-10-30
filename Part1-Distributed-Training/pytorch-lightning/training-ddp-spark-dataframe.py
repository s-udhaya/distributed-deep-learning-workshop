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
WORLD_SIZE = NUM_DEVICES * GPU_PER_DEVICE
BATCH_SIZE = 64

READER_POOL_TYPE = "thread"
WORKERS_COUNT = 2
MAX_EPOCHS = 500

# COMMAND ----------

from petastorm.spark import SparkDatasetConverter

dbutils.fs.rm("file:///dbfs/tmp/petastorm/cache2", True)

Data_Directory = '/databricks-datasets/flowers/delta'

CACHE_DIR = "file:///dbfs/tmp/petastorm/cache2"


spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, CACHE_DIR)

flowers_df, train_converter, val_converter = prepare_data(data_dir=Data_Directory, 
                                                          num_devices=WORLD_SIZE)


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

import os
import mlflow
from math import ceil
from spark_pytorch_distributor import MirrorRunner
from functools import partial


def train_mirrored():
    ## we pass the experiment id over to the workers
  train_dataloader_callable = partial(data_loader, train_converter)
  val_dataloader_callable = partial(data_loader, val_converter)

  # mlflow workaround to ensure that horovod subprocesses can find and connect to mlflow
  mlflow.set_tracking_uri("databricks")
  experiment = prepare_mlflow_experiment(username, DATABRICKS_HOST, DATABRICKS_TOKEN, "pytorch-lightning-petastorm-dataframe-load")
  print(f"node rank{int(os.environ['NODE_RANK'])}")
  ## we pass the experiment id over to the workers
  mlflow_experiment_id = experiment.experiment_id
  mlflow_experiment_id = experiment.experiment_id

  
  train_steps_per_epoch = ceil(len(train_converter) //  (BATCH_SIZE * WORLD_SIZE))
  val_steps_per_epoch = ceil(len(val_converter) //  (BATCH_SIZE * WORLD_SIZE)) 

  model = LitClassificationModel(class_count=5, learning_rate=1e-5*WORLD_SIZE, 
                                       device_id=int(os.environ["NODE_RANK"]), device_count=WORLD_SIZE, label_column="label")
  datamodule = FlowersDataModule(train_dataloader_callable=train_dataloader_callable, 
                                 val_dataloader_callable=val_dataloader_callable, reader_pool_type=READER_POOL_TYPE, device_id=int(os.environ["NODE_RANK"]), device_count=WORLD_SIZE, batch_size=BATCH_SIZE, workers_count=WORKERS_COUNT, label_column="label")
  default_dir = f'/dbfs/Users/{username}/tmp/lightning'

    # `gpus` parameter here should be 1 because the parallelism is controlled by Horovod
  return train(model, datamodule, gpus=1, num_nodes=WORLD_SIZE, strategy="ddp", device_id=int(os.environ["NODE_RANK"]), device_count=WORLD_SIZE, batch_size=BATCH_SIZE, 
                 mlflow_experiment_id=mlflow_experiment_id,
                default_dir=default_dir, train_steps_per_epoch=train_steps_per_epoch, val_steps_per_epoch=val_steps_per_epoch, workers_count=WORKERS_COUNT, 
        reader_pool_type=READER_POOL_TYPE, max_epochs=MAX_EPOCHS)

# COMMAND ----------

model = MirrorRunner(num_slots=WORLD_SIZE, use_custom_strategy=True, local_mode=False).run(train_mirrored)
