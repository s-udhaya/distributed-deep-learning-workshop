import datetime as dt
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
import mlflow

EARLY_STOP_MIN_DELTA = 0.01
EARLY_STOP_PATIENCE = 10

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
          device_count:int=1, batch_size:int=16, train_steps_per_epoch:int=1, val_steps_per_epoch:int=1, max_epochs:int=1000, workers_count: int = 1, reader_pool_type: str = "dummy", logging_level=logging.INFO,
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
    print(f"- training steps per epoch: {train_steps_per_epoch}")
    print(f"- validation steps per epoch: {val_steps_per_epoch}")
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
      limit_train_batches=train_steps_per_epoch,  # this is the way to end the epoch
      log_every_n_steps=1,
      val_check_interval=train_steps_per_epoch,  # this value must be the same as `limit_train_batches`
      num_sanity_val_steps=0,  # this must be zero to prevent a Petastorm error about Data Loader not being read completely
      limit_val_batches=val_steps_per_epoch,  # any value would work here but there is point in validating on repeated set of data
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
      mlflow.log_params({"workers_count": workers_count, "reader_pool_type": reader_pool_type, "batch_size": batch_size, "train_steps_per_epoch": train_steps_per_epoch, 
          "val_steps_per_epoch": val_steps_per_epoch})
      trainer.fit(model, dataloader, ckpt_path=ckpt_restore)
      report_duration(f"Training", start)
      print("\n\n---------------------")
  else:
    trainer.fit(model, dataloader, ckpt_path=ckpt_restore)
      
  
  return model.model if device_id == 0 else None