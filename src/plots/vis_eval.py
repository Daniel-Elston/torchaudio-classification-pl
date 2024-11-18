from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config.model import ModelConfig
from config.state_init import StateManager

class VisualiseEvaluation:
    def __init__(self, state: StateManager, config: ModelConfig):
        self.config = config
        self.model_state = state.model_state
        
    def pipeline(self):
        self.plot_report()
        self.plot_loss()
        self.vis_cm()

    def plot_report(self):
        report = self.model_state.get("report")
        
        print(report.keys())

    def plot_loss(self):
        train_losses = self.model_state.get('train_losses')
        val_losses = self.model_state.get('val_losses')
        print(val_losses)
        print('\n')
        print(train_losses)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Losses Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    def vis_cm(self):
        cm = self.model_state.get("cm")
        if isinstance(cm, dict):  # If stored as a dict, convert back to DataFrame
            cm = pd.DataFrame.from_dict(cm)
        print(cm)
        print(cm.columns)
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()
        
    def __call__(self):
        self.pipeline()