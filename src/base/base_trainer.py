from __future__ import annotations


import pytorch_lightning as pl
import torch

class BaseTrainer(pl.LightningModule):
    def __init__(self, model, config):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.config = config
        self.criterion = self.configure_criterion()

    def forward(self, x):
        return self.model(x)

    def configure_criterion(self):
        """Define the loss function."""
        return torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        inputs, labels = self.process_batch(batch)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = self.process_batch(batch)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = self.process_batch(batch)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer

    def process_batch(self, batch):
        inputs = batch['spectrogram']
        labels = batch['label']
        return inputs, labels

