# Databricks notebook source
# MAGIC %run ../00_setup 

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

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

NUM_DEVICES = 2
BATCH_SIZE = 64

READER_POOL_TYPE = "thread"
WORKERS_COUNT = 2

# COMMAND ----------

from petastorm.spark import SparkDatasetConverter
from src.data_preprocessor import prepare_data

dbutils.fs.rm("file:///dbfs/tmp/petastorm/cache", True)

Data_Directory = '/databricks-datasets/flowers/delta'

CACHE_DIR = "file:///dbfs/tmp/petastorm/cache"


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

from math import ceil
import mlflow
from functools import partial
from src.data_module import FlowersDataModule
from src.model import LitClassificationModel
from src.training import train
from src.mlflow_util import prepare_mlflow_experiment


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

def get_model(hvd_size, hvd_rank):
  from src.model import LitClassificationModel
  return LitClassificationModel(class_count=5, learning_rate=1e-5*hvd_size, 
                                     device_id=hvd_rank, device_count=hvd_size, label_column="label")

def get_data_module(hvd_size, hvd_rank, batch_size):
  from src.data_module import FlowersDataModule
  train_dataloader_callable = partial(data_loader, train_converter)
  val_dataloader_callable = partial(data_loader, val_converter)
  return FlowersDataModule(train_dataloader_callable=train_dataloader_callable, 
                               val_dataloader_callable=val_dataloader_callable, reader_pool_type=READER_POOL_TYPE, label_column="label", device_id=hvd_rank, device_count=hvd_size, batch_size=batch_size, workers_count=WORKERS_COUNT)

def get_train():
  from src.training import train
  return train

# COMMAND ----------

import mlflow
from functools import partial
from src.mlflow_util import prepare_mlflow_experiment

experiment = prepare_mlflow_experiment(username, DATABRICKS_HOST, DATABRICKS_TOKEN, "pytorch-lightning-petastorm-dataframe-load")

def train_hvd():
  from math import ceil
  import sys
  sys.path.append('/Workspace/Repos/udhayaraj.sivalingam@databricks.com/distributed-deep-learning-workshop/Part1-Distributed-Training/pytorch-lightning/src')
  hvd.init()
  import sys
  print(sys.path)

  print("importing src")

  # mlflow workaround to ensure that horovod subprocesses can find and connect to mlflow
  mlflow.set_tracking_uri("databricks")
  
  ## we pass the experiment id over to the workers
  mlflow_experiment_id = experiment.experiment_id
  batch_size = BATCH_SIZE * NUM_DEVICES
  train_steps_per_epoch = ceil(len(train_converter) //  batch_size)
  val_steps_per_epoch = ceil(len(val_converter) //  batch_size)  
  hvd_model = get_model(hvd_size=hvd.size(),hvd_rank=hvd.rank())
  
  hvd_datamodule = get_data_module(hvd_size=hvd.size(),hvd_rank=hvd.rank(), batch_size=batch_size)
  default_dir = f'/dbfs/Users/{username}/tmp/lightning'

  # `gpus` parameter here should be 1 because the parallelism is controlled by Horovod
  return get_train(hvd_model, hvd_datamodule, gpus=1, device_id=hvd.rank(), device_count=hvd.size(), batch_size=batch_size, 
               mlflow_experiment_id=mlflow_experiment_id,
              default_dir=default_dir, train_steps_per_epoch=train_steps_per_epoch, val_steps_per_epoch=val_steps_per_epoch, workers_count=WORKERS_COUNT, 
      reader_pool_type=READER_POOL_TYPE, max_epochs=1000)

# COMMAND ----------

from sparkdl import HorovodRunner

# # with a negative np number, we will launch multi gpu training on one node
hr = HorovodRunner(np=-1, driver_log_verbosity='all')
hvd_model = hr.run(train_hvd)

# COMMAND ----------

# MAGIC %run ./src/__init__.py

# COMMAND ----------

from sparkdl import HorovodRunner


# This will launch a distributed training on np devices
hr = HorovodRunner(np=2, driver_log_verbosity='all')
hvd_model = hr.run(train_hvd)
