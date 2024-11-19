from __future__ import annotations

import json
import logging
from pprint import pformat

from torch.utils.data import DataLoader
from config.state_init import StateManager
from torchvision import datasets, transforms
import torch
from collections import Counter
from config.data import DataConfig
from config.model import ModelConfig
from utils.file_access import FileAccess
from src.data.make_dataset import HDF5AudioDataset
import pytorch_lightning as pl
from torch.utils.data import random_split, Subset


class HDF5DataModule(pl.LightningDataModule):
    def __init__(
        self, config: ModelConfig, hdf5_filename,
        num_workers=4, subset=False, label_to_idx=None
    ):
        super().__init__()
        self.config = config
        self.hdf5_filename = hdf5_filename
        self.num_workers = num_workers
        self.subset = subset
        self.label_to_idx = label_to_idx

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        full_dataset = HDF5AudioDataset(
            self.hdf5_filename,
            transform=None,
            label_to_idx=self.label_to_idx
        )
        
        if self.subset:
            subset_indices = range(int(len(full_dataset) * 0.5))
            full_dataset = Subset(full_dataset, subset_indices)
            logging.debug(f"Using subset of dataset for testing: {len(full_dataset)} samples")
        
        total_size = len(full_dataset)
        train_size = int(self.config.train_size * total_size)
        val_size = int(self.config.val_size * total_size)
        test_size = total_size - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=self.num_workers
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.num_workers
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.num_workers
        )
