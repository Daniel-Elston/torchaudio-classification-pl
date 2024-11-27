from __future__ import annotations

import json
from config.state_init import StateManager
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.models.cnn import CNN
from src.base.base_trainer import BaseTrainer
from src.data.data_module import HDF5DataModule
from config.model import ModelConfig
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.metrics import MetricLogger
from src.plots.vis_eval import VisualiseEvaluation


class ModelTrainer:
    def __init__(
        self, state: StateManager,
        config: ModelConfig,
        data_module: HDF5DataModule,
        model: CNN,
        trainer_model: BaseTrainer
    ):
        self.state = state
        self.config = config
        self.data_module = data_module
        self.model = model
        self.trainer_model = trainer_model
        self.metric_logger = MetricLogger(num_classes=self.config.num_classes)
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
            callbacks=[checkpoint_callback, early_stopping_callback, self.metric_logger],
            accelerator='cuda' if self.config.device == 'cuda' else 'cpu',
            logger=logger,
            enable_model_summary=True,
            enable_progress_bar=True,
            enable_checkpointing=True
        )
        return trainer

    def train(self):
        self.trainer.fit(self.trainer_model, datamodule=self.data_module)
    
    def validate(self):
        self.trainer.validate(self.trainer_model, datamodule=self.data_module)

    def test(self):
        self.trainer.test(self.trainer_model, datamodule=self.data_module)
    
    def visualize_evaluation(self):
        visualizer = VisualiseEvaluation(
            train_losses=self.metric_logger.train_loss_list,
            val_losses=self.metric_logger.val_loss_list,
            train_accuracies=self.metric_logger.train_acc_list,
            val_accuracies=self.metric_logger.val_acc_list,
            val_precisions=self.metric_logger.val_precision_list,
            val_recalls=self.metric_logger.val_recall_list,
            val_f1s=self.metric_logger.val_f1_list,
            confusion_matrix=self.metric_logger.confusion_matrix
        )
        visualizer()

    def run(self):
        self.train()
        # self.validate()
        self.test()
        self.visualize_evaluation()
        
    def __call__(self):
        self.run()

