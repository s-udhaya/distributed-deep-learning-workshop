from memory_profiler import profile

@profile
def run_memory_profiled_training():
  BATCH_SIZE = 64
  STEPS_PER_EPOCH = train_rows //  BATCH_SIZE
  datamodule = FlowersDataModule(train_parquet_files=train_parquet_files, 
                                 val_parquet_files=val_parquet_files, batch_size=BATCH_SIZE, workers_count=2)
  model = LitClassificationModel(class_count=5, learning_rate=1e-5)

  train(model, datamodule, gpus=1, default_dir=default_dir, batch_size=BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH)
  
if __name__ == '__main__':
    run_memory_profiled_training()