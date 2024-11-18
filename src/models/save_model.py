from __future__ import annotations

import logging
from config.model import ModelConfig
from config.state_init import StateManager
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.cnn import CNN


class SaveModel:
    def __init__(self, state: StateManager, config: ModelConfig):
        self.config = config
        self.model_state = state.model_state
        self.save_path = state.paths.get_path('models')
    
    def __call__(self):
        model = self.model_state.get('model')
        optimizer = self.model_state.get('optimizer')
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
            'val_accuracy': self.model_state.get('val_accuracy'),
            'val_loss': self.model_state.get('val_loss')
        }, self.save_path)
        logging.info(f"Model and optimizer states saved to: {self.save_path}")
