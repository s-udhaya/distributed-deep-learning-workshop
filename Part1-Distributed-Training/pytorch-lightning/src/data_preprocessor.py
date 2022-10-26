from petastorm.spark import SparkDatasetConverter, make_spark_converter
import pyspark.sql.types as T
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
import pyspark.sql.types as T
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession


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
    spark = SparkSession.builder.getOrCreate()
    
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