from __future__ import annotations

import logging
from config.model import ModelConfig
from config.state_init import StateManager
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.cnn import CNN
from src.base.base_val import BaseValidator


class TrainModel:
    def __init__(self, state: StateManager, config: ModelConfig):
        self.config = config
        self.model_state = state.model_state
        self.data_state = state.data_state

    def __call__(self):
        model = self.model_state.get("model")
        train_loader = self.data_state.get("train_dataloader")
        val_loader = self.data_state.get("val_dataloader")  # Ensure you have a validation dataloader
        device = self.config.device
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        num_epochs = self.config.epochs

        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            self.model_state.set('train_losses', train_losses)

            # Perform validation after each epoch using BaseValidator
            validator = BaseValidator(model, val_loader, device)
            avg_val_loss, val_accuracy = validator.validate(criterion=criterion)

            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            self.model_state.set('val_losses', val_losses)
            self.model_state.set('val_accuracies', val_accuracies)

            print(
                f"Epoch {epoch+1}/{num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.2%}")
