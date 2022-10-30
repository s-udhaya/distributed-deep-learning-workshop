# Databricks notebook source
# MAGIC %md
# MAGIC # 02. Model Training - Single node 

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook we will be performing some basic transfer learning on the flowers dataset, using the [`MobileNetV2`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/MobileNetV2) model as our base model.
# MAGIC 
# MAGIC We will:
# MAGIC - Load the train and validation datasets created in [01_data_prep]($./01_data_prep), converting them to `torch.utils.data.Dataset` objects. 
# MAGIC - Train our model on a single (driver) node.

# COMMAND ----------

# MAGIC %run ../../Part0-setup-and-data-preparation/00_setup
# MAGIC %run ./source_modules

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
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torchvision import transforms

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
# MAGIC Load the train and validation datasets we created in [01_data_prep]($./01_data_prep) as Spark DataFrames.

# COMMAND ----------

cols_to_keep = ['content', 'label_idx']

train_tbl_name = 'silver_train'
val_tbl_name = 'silver_val'

train_df = (spark.table(f'{database_name}.{train_tbl_name}')
            .select(cols_to_keep))

val_df = (spark.table(f'{database_name}.{val_tbl_name}')
          .select(cols_to_keep))

print('train_df count:', train_df.count())
print('val_df count:', val_df.count())

# COMMAND ----------

num_classes = train_df.select('label_idx').distinct().count()

print('Number of classes:', num_classes)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## iii. Create `torch.utils.data.Dataset` and `DataLoader`

# COMMAND ----------

# MAGIC %md
# MAGIC To train a model with pytorch, we'll need to transform our spark dataframes into `torch.utils.data.Dataset` objects. 
# MAGIC 
# MAGIC For this example, we'll convert our dataframe to a pandas dataframe using `.toPandas()` and convert to a properly formated `torch.utils.data.Dataset` from there.

# COMMAND ----------

train_pdf = train_df.toPandas() 
val_pdf = val_df.toPandas()



train_data = FlowerDataset(train_pdf, transform)
val_data = FlowerDataset(val_pdf, transform)


train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## iv. Define model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Single node training is the most common way machine learning practitioners set up their training execution. For some modeling use cases, this is a great way to go as all the modeling can stay on one machine with no additional libraries. 
# MAGIC 
# MAGIC Training with Databricks is just as easy. Simply use a [Single Node Cluster](https://docs.databricks.com/clusters/single-node.html) which is just a Spark Driver with no worker nodes. 
# MAGIC 
# MAGIC We will be using the [`MobileNetV2`](https://pytorch.org/vision/main/models/mobilenetv2.html) architecture from torchvision as our base model, freezing the base layers and adding a
# MAGIC Dense classification layer.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## v. Train Model - Single Node

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's train out model, where we will instantiate and compile the model, and fit the model; using [MLflow autologging](https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/databricks-autologging) to automatically track model params, metrics and artifacts.

# COMMAND ----------



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
  

  for epoch in range(EPOCHS):
    print('Epoch {}/{}'.format(epoch + 1, EPOCHS))
    print('-' * 10)

    train_loss, train_acc = train_one_epoch(model, criterion, optimizer, exp_lr_scheduler, 
                                            train_dataloader, epoch, 
                                            device)
    val_loss, val_acc = evaluate(model, criterion, val_dataloader, device)

  return val_loss
  
loss = train_and_evaluate()

# COMMAND ----------

# MAGIC %md
# MAGIC Wohoo! We trained a model on a single machine very similarly to what you might have done on your laptop. It was easy for us to convert this small dataset into a pandas DataFrame from a Delta table and subsequently convert it into a `tf.data.Dataset` for training. 
# MAGIC 
# MAGIC However, most production workloads will require training with **orders of magnitude** more data, so much so that it could overwhelm a single machine while training. Other cases could contribute to exhausting a single node during training such as training large model architectures.
# MAGIC 
# MAGIC In the next notebook, [03_model_training_distributed]($./03_model_training_distributed) we will see how we scale our model training across mutliple nodes.
