from __future__ import annotations

import json
from config.state_init import StateManager
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.models.cnn import CNN
from src.base.base_trainer import BaseTrainer
from src.data.load_imgs import HDF5DataModule
from config.model import ModelConfig
from pytorch_lightning.loggers import TensorBoardLogger

def load_label_mapping():
    with open('reports/label_mapping.json', 'r') as f:
        label_to_idx = json.load(f)
        return label_to_idx


class ModelTrainer:
    def __init__(self, state: StateManager):
        self.state = state
        self.config = ModelConfig()
        self.model = CNN(self.config)
        self.data_module = HDF5DataModule(
            config=self.config,
            hdf5_filename='data/audio_data.hdf5',
            subset=self.config.subset,
            label_to_idx=load_label_mapping(),
        )
        self.trainer_model = BaseTrainer(
            self.model, 
            self.config, 
        )
        self.trainer = self.configure_trainer()

    def configure_trainer(self):
        logger = TensorBoardLogger("tb_logs", name="audio_classification")
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints/',
            filename='best-checkpoint',
            save_top_k=1,
            mode='min'
        )

        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min'
        )

        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            callbacks=[checkpoint_callback, early_stopping_callback],
            accelerator='cuda' if self.config.device == 'cuda' else 'cpu',
            logger=logger,
            enable_model_summary=True,
            enable_progress_bar=True
        )
        return trainer

    def train(self):
        self.trainer.fit(self.trainer_model, datamodule=self.data_module)

    def test(self):
        self.trainer.test(self.trainer_model, datamodule=self.data_module)

    def run(self):
        self.train()
        self.test()
        
    def __call__(self):
        self.run()

if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.run()
