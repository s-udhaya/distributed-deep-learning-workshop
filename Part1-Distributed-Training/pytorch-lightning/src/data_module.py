import pytorch_lightning as pl
from petastorm import TransformSpec
from PIL import Image
from torchvision import transforms
import numpy as np
import io


class FlowersDataModule(pl.LightningDataModule):

    def __init__(self, train_dataloader_callable, val_dataloader_callable, device_id: int = 0, device_count: int = 1, batch_size: int = 16,
                 workers_count: int = 1, reader_pool_type: str = "dummy", feature_column: str = "content", label_column: str = "label_idx"):

        self.train_dataloader_callable = train_dataloader_callable
        self.val_dataloader_callable = val_dataloader_callable
        self.train_dataloader_context = None
        self.val_dataloader_context = None
        self.prepare_data_per_node = False
        self._log_hyperparams = False
        self.device_id = device_id
        self.device_count = device_count
        self.batch_size = batch_size
        self.workers_count = workers_count
        self.reader_pool_type = reader_pool_type
        self.feature_column=feature_column
        self.label_column=label_column

    def train_dataloader(self):
        if self.train_dataloader_context is not None:
          self.train_dataloader_context.__exit__(None, None, None)
        self.train_dataloader_context = self.train_dataloader_callable(
                transform_spec=self._get_transform_spec(),
                num_epochs=None,
                workers_count=self.workers_count,
                cur_shard=self.device_id,
                shard_count=self.device_count,
              reader_pool_type=self.reader_pool_type,
                batch_size=self.batch_size)
        return self.train_dataloader_context.__enter__()

    def val_dataloader(self):
        if self.val_dataloader_context is not None:
          self.val_dataloader_context.__exit__(None, None, None)
        self.val_dataloader_context = self.val_dataloader_callable(
            transform_spec=self._get_transform_spec(),
            num_epochs=None,
            workers_count=self.workers_count,
            cur_shard=self.device_id,
            shard_count=self.device_count,
          reader_pool_type=self.reader_pool_type,
            batch_size=self.batch_size)
        return self.val_dataloader_context.__enter__()

    def teardown(self, stage=None):
        # Close all readers (especially important for distributed training to prevent errors)
        self.train_dataloader_context.__exit__(None, None, None)
        self.val_dataloader_context.__exit__(None, None, None)

    def preprocess(self, img):

        image = Image.open(io.BytesIO(img))
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return transform(image)

    def _transform_rows(self, batch):
        batch = batch[[self.feature_column, self.label_column]]
        # To keep things simple, use the same transformation both for training and validation
        batch["features"] = batch[self.feature_column].map(lambda x: self.preprocess(x).numpy())
        batch = batch.drop(labels=[self.feature_column], axis=1)
        return batch

    def _get_transform_spec(self):
        return TransformSpec(self._transform_rows,
                             edit_fields=[("features", np.float32, (3, 224, 224), False)],
                             selected_fields=["features", self.label_column])
