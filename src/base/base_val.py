from __future__ import annotations

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class BaseValidator:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def validate(self, criterion=None, compute_metrics=False):
        """Perform validation and compute metrics."""
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # Compute loss if criterion is provided
                if criterion:
                    loss = criterion(output, target)
                    total_loss += loss.item()

                _, predicted = torch.max(output, 1)
                total_correct += (predicted == target).sum().item()
                total_samples += target.size(0)

                if compute_metrics:
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

        # Calculate average loss and accuracy
        avg_loss = total_loss / len(self.dataloader) if criterion else None
        accuracy = total_correct / total_samples

        if compute_metrics:
            return np.array(all_preds), np.array(all_targets), avg_loss, accuracy
        else:
            return avg_loss, accuracy
