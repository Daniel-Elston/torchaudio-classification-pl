from __future__ import annotations

import logging
from config.state_init import StateManager
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.cnn import CNN
from config.model import ModelConfig


class InitialiseModel:
    def __init__(self, state: StateManager, config: ModelConfig):
        self.model_state = state.model_state
        self.config = config
    
    def __call__(self):
        model = CNN(self.config).to(self.config.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        self.model_state.set('model', model)
        self.model_state.set("optimizer", optimizer)

        logging.info(
            f"Model created and initialised: {model.__class__.__name__} and stored in {self.model_state.__class__.__name__}")
        logging.info(
            f"Optimizer created and initialised: {model.__class__.__name__} and stored in {self.model_state.__class__.__name__}")
        return model
