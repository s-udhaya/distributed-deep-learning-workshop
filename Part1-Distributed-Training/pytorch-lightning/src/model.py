import pytorch_lightning as pl



class LitClassificationModel(pl.LightningModule):
  """
  
  Our main model class
  
  
  """
  
  def __init__(self, class_count: int, learning_rate:float, momentum:float=0.9, logging_level=logging.INFO,
              device_id:int=0, device_count:int=1, family:str='mobilenet', feature_column: str = "features", label_column: str = "label"):
    
    super().__init__()
    self.learn_rate = learning_rate
    self.momentum = momentum
    self.model = self.get_model(class_count, learning_rate, family)
    self.state = {"epochs": 0}
    self.logging_level = logging_level
    self.device_id = device_id
    self.device_count = device_count
    self.family = family
    self.feature_column = feature_column
    self.label_column = label_column
  
  def get_model(self, class_count, lr, family):
    """
    
    This is the function that initialises our model.
    If we wanted to use other prebuilt model libraries like timm we would put that model here
    
    """
    
    if family == 'mobilenet':
      model = models.mobilenet_v2(pretrained=True)
    elif family == 'resnext':
      model = models.resnext50_32x4d(pretrained=True)
    
    # Freeze parameters in the feature extraction layers and replace the last layer
    for param in model.parameters():
      param.requires_grad = False

    # New modules have `requires_grad = True` by default
    if family == 'mobilenet':
      model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, class_count)
    elif family == 'resnext':
      model.fc = torch.nn.Linear(model.fc.in_features, class_count)
    
    
    return model
  
  def configure_optimizers(self):
    
    if self.family == 'mobilenet':
      params = self.model.classifier[1].parameters()
    elif self.family == 'resnext':
      params = self.model.fc.parameters()
    
    optimizer = torch.optim.SGD(params, lr=self.learn_rate, momentum=self.momentum)
    
    return optimizer
  
  def forward(self, inputs):
    outputs = self.model(inputs)
    
    return outputs
  
  def training_step(self, batch, batch_idx):
    X, y = batch[self.feature_column], batch[self.label_column].type(torch.LongTensor).to(device)
    pred = self(X)
    loss = F.cross_entropy(pred, y)
    
    # Choosing to use step loss as a metric
    self.log("train_loss", loss, prog_bar=True)
    
    if self.logging_level == logging.DEBUG:
      if batch_idx == 0:
        print(f" - [{self.device_id}] training batch size: {y.shape[0]}")
      print(f" - [{self.device_id}] training batch: {batch_idx}, loss: {loss}")
      
    return loss
  
  def on_train_epoch_start(self):
    # No need to re-load data here as `train_dataloader` will be called on each epoch
    if self.logging_level in (logging.DEBUG, logging.INFO):
      print(f"++ [{self.device_id}] Epoch: {self.state['epochs']}")
    self.state["epochs"] += 1
    
  def validation_step(self, batch, batch_idx):
    X, y = batch[self.feature_column], batch[self.label_column].type(torch.LongTensor).to(device)
    pred = self(X)
    loss = F.cross_entropy(pred, y)
    acc = FM.accuracy(pred, y)

    # Roll validation up to epoch level
    self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
    self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
    
    if self.logging_level == logging.DEBUG:
      print(f" - [{self.device_id}] val batch: {batch_idx}, size: {y.shape[0]}, loss: {loss}, acc: {acc}")

    return {"loss": loss, "acc": acc}