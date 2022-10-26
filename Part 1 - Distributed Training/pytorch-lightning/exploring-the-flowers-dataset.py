# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Exploring the Flowers Dataset and Petastorm
# MAGIC 
# MAGIC In this exercise, we will show how to work with Parquet image datasets and also how to convert data into Parquet format

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Setup
# MAGIC 
# MAGIC The notebook can be run on a non-GPU standard Databricks ML Runtime Cluster.
# MAGIC 
# MAGIC It has been tested with both Databricks Runtime 10.4 ML LTS and 11.1 ML with and without GPU
# MAGIC 
# MAGIC The Flowers dataset comes as one of the Databricks provided demo datasets in the */databricks-datasets/* folder.
# MAGIC The normal unprocessed dataset is also available in the folder */databricks-datasets/flower_photos*

# COMMAND ----------

# list the contents of the flowers dataset folder
display(dbutils.fs.ls('/databricks-datasets/flowers/delta'))

# COMMAND ----------

# list the contents of the flowers dataset folder
display(dbutils.fs.ls('/databricks-datasets/flower_photos'))

# COMMAND ----------

## Import Libraries
from petastorm.spark import SparkDatasetConverter, make_spark_converter
import pyspark.sql.types as T
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from petastorm import TransformSpec
import numpy as np
from PIL import Image
import io
from torchvision import transforms

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Config Parameters

# COMMAND ----------

### The default directory for the flowers dataset preconverted into the Delta format
DATA_DIR = "/databricks-datasets/flowers/delta"

SAMPLE_SIZE = 1000
print(f"Sample: size {SAMPLE_SIZE}")

## Petastorm requires an intermediate cache directory in order to store processed results
# Set a cache directory on DBFS FUSE for intermediate data.
CACHE_DIR = "file:///dbfs/tmp/petastorm/cache"
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, CACHE_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Loading the Delta Dataset

# COMMAND ----------

### Review the schema of the flowers dataset

flowers_df = spark.read.format("delta").load(DATA_DIR)

flowers_df.printSchema()

# COMMAND ----------

## The main columns of importance to us in this dataset is the label, the category of the flower, and the content, the image in parquet column format
# We can view the dataset with the display option
training_set = flowers_df.select('label', 'content') 

display(training_set)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Understanding Petastorm
# MAGIC 
# MAGIC Petastorm works as a converter library in order to make parquet datasets consumable in a standard deep learning data loader object.
# MAGIC For PyTorch, Petastorm provides an adapter into the `torch.utils.data.DataLoader` format
# MAGIC 
# MAGIC Note that the supported datatypes in Spark and PyTorch are different so there can be conversion errors along the way. In this case, PyTorch tensor does not support string object so we need to convert the `Label` to a numeric first. 

# COMMAND ----------

# Collect the set of string labels, andconvert the strings to numeric indices
classes = list(training_set.select("label").distinct().toPandas()["label"])

def to_class_index(class_name:str):
  """
  Converts classes to a class_index so that we can create a tensor object
  
  Args:
    class_name: the string form of a the classname
    
  Returns:
    the numeric index for a string class
  
  """
  
  return classes.index(class_name)


class_index = udf(to_class_index, T.LongType())  # PyTorch loader required a long

preprocessed_df = training_set.withColumn("label", class_index(col("label")))

display(preprocessed_df)

# COMMAND ----------

# to load a dataset into a deep learning framework we first need to push the spark frame through a converter

peta_conv_df = make_spark_converter(preprocessed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In Spark, the image is stored as a raw binary so in order for us to convert it into a tensor in our torch dataloader we need to write a decoder function first. This is also where we might insert logic from data augmentation libraries like ablumentations

# COMMAND ----------

def preprocess(img):
  """
  
  Args:
    img: binary image object
  Returns:
    Image transformed into a tensor
  
  """
  image = Image.open(io.BytesIO(img))
  transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  return transform(image)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The transformation needs to be applied through the make_torch_dataloader this is done via a transform spec. We also need to specify the output format for a transform spec

# COMMAND ----------

def transform_rows(batch: dict):
  
    """
    
    This function receives a batch of entries in dict form that need to be processed.
    In this case, our our processing function will open the image perform a some transfomations
    then turn it into a numeric tensor
    
    Args:
      batch: dict
        input batch
    
    Returns:
      batch: dict
        transformed batch 
    
    """
    
    # To keep things simple, use the same transformation both for training and validation
    batch["features"] = batch["content"].map(lambda x: preprocess(x).numpy())
    batch = batch.drop(labels=["content"], axis=1)
    return batch

transform_func = TransformSpec(transform_rows, 
                         edit_fields=[("features", np.float32, (3, 224, 224), False)], 
                         selected_fields=["features", "label"])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now we can apply the transform_func to the dataset in order to create batches.
# MAGIC The `converted_dataset` will be a valid dataloader to plug into our PyTorch dataloader

# COMMAND ----------

with peta_conv_df.make_torch_dataloader(transform_spec=transform_func) as converted_dataset:
  
  sample = next(iter(converted_dataset))
  print(sample)
  

# COMMAND ----------


