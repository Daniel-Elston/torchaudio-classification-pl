from __future__ import annotations

import logging
from config.model import ModelConfig
from config.state_init import StateManager
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.cnn import CNN


class LoadModel:
    def __init__(self, state: StateManager, config: ModelConfig, view: bool = False):
        self.config = config
        self.model_state = state.model_state
        self.load_path = state.paths.get_path('models')
        self.view = view

    def __call__(self):
        model = CNN(self.config)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        checkpoint = torch.load(self.load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        model.to(self.config.device)
        
        self.model_state.set("model", model)
        self.model_state.set("optimizer", optimizer)
        logging.info(f"Model and optimizer loaded from {self.load_path}")
        
        if self.view:
            self._view_model()

    def _view_model(self):
        model = self.model_state.get("model")
        optimizer = self.model_state.get("optimizer")

        logging.info(f"Model: {model}")
        # logging.info(f"Model parameters: {model.parameters()}")
        # logging.info(f"Model state dict: {model.state_dict()}")
        

        # logging.warning("Model's state_dict:")
        # for param_tensor in model.state_dict():
        #     logging.info(param_tensor, "\t", model.state_dict()[param_tensor].size())

        # logging.warning("Optimizer's state_dict:")
        # for var_name in optimizer.state_dict():
        #     logging.info(var_name, "\t", optimizer.state_dict()[var_name])
