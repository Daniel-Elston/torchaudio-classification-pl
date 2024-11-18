from __future__ import annotations

import json
import logging
from config.model import ModelConfig
from config.state_init import StateManager
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.cnn import CNN
from sklearn.metrics import classification_report, confusion_matrix
from utils.file_access import FileAccess, load_json
import pandas as pd
from pprint import pprint
# from src.models.validate import ValidateModel
from src.base.base_val import BaseValidator


class EvaluateModel:
    def __init__(self, state: StateManager, config: ModelConfig):
        self.config = config
        self.model_state = state.model_state
        self.data_state = state.data_state
        self.save_path = state.paths.get_path("results")

    def __call__(self):
        model = self.model_state.get("model")
        test_loader = self.data_state.get("test_dataloader")
        device = self.config.device
        
        validator = BaseValidator(model, test_loader, device)
        all_preds, all_targets, _, accuracy = validator.validate()
        
        report = classification_report(all_targets, all_preds, output_dict=True)
        cm = confusion_matrix(all_targets, all_preds)
        report, cm_df = self._apply_inv_map(report, cm)
        
        self._save_metrics_to_state(accuracy, report, cm_df)
        self._save_metrics_to_file(accuracy, report, cm_df)

    def _apply_inv_map(self, report, cm):
        mapping = load_json('reports/mappings.json')
        print('MAPPING:', mapping)
        inverse_mapping = {v: k for k, v in mapping.items()}
        
        mapped_report = {
            (inverse_mapping[int(k)] if k.isdigit() else k): v
            for k, v in report.items()
        }
        
        cm_df = pd.DataFrame(
            cm,
            index=[inverse_mapping[i] for i in range(len(cm))],
            columns=[inverse_mapping[i] for i in range(len(cm))]
        )
        return mapped_report, cm_df

    def _save_metrics_to_state(self, acc, report, cm_df):
        self.model_state.set("acc", acc)
        self.model_state.set("report", report)
        self.model_state.set("cm", cm_df)

    def _save_metrics_to_file(self, acc, report, cm_df):
        results = {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm_df,
        }
        FileAccess.save_json(results, self.save_path)

