import os
import mlflow


def prepare_mlflow_experiment(username, databricks_host, databricks_token, experiment_name):
  # When using databricks repos, it is not possible to write into working directories
  # specifying a dbfs default dir helps to avoid this
  experiment_path = f'/Users/{username}/{experiment_name}'
  ## Specifying the mlflow host server and access token 
  # We put them to a variable to feed into horovod later on
  # We put them into these environment variables as this is where mlflow will look by default
  os.environ['DATABRICKS_HOST'] = databricks_host
  os.environ['DATABRICKS_TOKEN'] = databricks_token
  # We manually create the experiment so that we know the id and can send that to the worker nodes when we scale
  experiment = mlflow.set_experiment(experiment_path)
  return experiment