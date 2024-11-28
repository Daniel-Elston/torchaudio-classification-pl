from __future__ import annotations

import logging

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from config.model import ModelConfig
from src.data.make_dataset import HDF5AudioDataset


class HDF5DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: ModelConfig,
        dataset: HDF5AudioDataset,
        hdf5_filename,
        num_workers=4,
        subset=False,
    ):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.hdf5_filename = hdf5_filename
        self.num_workers = num_workers
        self.subset = subset

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.subset:
            subset_indices = range(int(len(self.dataset) * 0.5))
            self.dataset = Subset(self.dataset, subset_indices)
            logging.debug(f"Using subset of dataset for testing: {len(self.dataset)} samples")

        total_size = len(self.dataset)
        train_size = int(self.config.train_size * total_size)
        val_size = int(self.config.val_size * total_size)
        test_size = total_size - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
