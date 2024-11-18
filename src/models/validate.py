from __future__ import annotations

from config.model import ModelConfig
from config.state_init import StateManager

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.base.base_val import BaseValidator


class ValidateModel:
    def __init__(self, state: StateManager, config: ModelConfig):
        self.config = config
        self.model_state = state.model_state
        self.data_state = state.data_state

    def __call__(self):
        model = self.model_state.get("model")
        val_loader = self.data_state.get("test_dataloader")
        device = self.config.device

        # Initialize BaseValidator
        validator = BaseValidator(model, val_loader, device)

        # Perform validation
        criterion = torch.nn.CrossEntropyLoss()
        all_preds, all_targets, avg_loss, accuracy = validator.validate(criterion)

        # Save metrics to model state
        val_losses = self.model_state.get("val_losses")
        val_losses.append(avg_loss)

        # Save metrics to model state
        self.model_state.set("val_losses", val_losses)
        self.model_state.set("val_accuracy", accuracy)

        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
