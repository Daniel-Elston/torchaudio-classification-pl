from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from config.model import ModelConfig
from config.state_init import StateManager
from src.base.base_trainer import BaseTrainer
from src.data.data_module import HDF5DataModule
from src.models.cnn import CNN
from src.models.metrics import MetricLogger


class ModelTrainer:
    def __init__(
        self,
        state: StateManager,
        config: ModelConfig,
        data_module: HDF5DataModule,
        model: CNN,
        trainer_model: BaseTrainer,
        metric_logger: MetricLogger,
    ):
        self.state = state
        self.config = config
        self.data_module = data_module
        self.model = model
        self.trainer_model = trainer_model
        self.idx_to_label = data_module.dataset.idx_to_label
        self.metric_logger = metric_logger
        self.trainer = self.configure_trainer()

    def __call__(self):
        self.train()
        self.validate()
        self.test()

    def configure_trainer(self):
        logger = TensorBoardLogger("tb_logs", name="audio_classification")
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints/",
            filename="best-checkpoint",
            save_top_k=1,
            mode="min",
        )

        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            callbacks=[checkpoint_callback, early_stopping_callback, self.metric_logger],
            accelerator="cuda" if self.config.device == "cuda" else "cpu",
            logger=logger,
            enable_model_summary=True,
            enable_progress_bar=True,
            enable_checkpointing=True,
        )
        return trainer

    def train(self):
        self.trainer.fit(self.trainer_model, datamodule=self.data_module)

    def validate(self):
        self.trainer.validate(self.trainer_model, datamodule=self.data_module)

    def test(self):
        self.trainer.test(self.trainer_model, datamodule=self.data_module)
