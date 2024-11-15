from __future__ import annotations

import logging
from pprint import pformat

from torch.utils.data import DataLoader
from config.state_init import StateManager
from torchvision import datasets, transforms
import torch
from collections import Counter


class LoadImages:
    def __init__(self, state: StateManager, img_type='spectograms', batch_size=64, view=None):
        self.data_state = state.data_state
        self.batch_size = batch_size
        self.img_type = img_type
        self.load_path = f'data/processed/{self.img_type}'
        self.train_dataloader = None
        self.test_dataloader = None
        self.view = view

    def run(self):
        train_dataset, test_dataset = self.create_dataset()
        
        train_classes = [label for _, label in train_dataset]
        logging.debug(f"Train class distribution: {Counter(train_classes)}")
        
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            
        self.data_state.set('train_dataloader', self.train_dataloader)
        self.data_state.set('test_dataloader', self.test_dataloader)
        logging.debug(
            f"Train dataloader created: {self.train_dataloader.__class__.__name__} and stored in {self.data_state.__class__.__name__}")
        logging.debug(
            f"Test dataloader created: {self.test_dataloader.__class__.__name__} and stored in {self.data_state.__class__.__name__}")
        
        if self.view:
            self._view_loaders()

    def create_dataset(self):
        dataset = datasets.ImageFolder(
            root=self.load_path,
            transform=transforms.Compose([
                transforms.Resize((200,80)),
                transforms.ToTensor()
            ])
        )
        class_to_idx = dataset.class_to_idx
        logging.debug(f"Dataset created: {dataset}")
        logging.debug(f"Dataset mapping: {pformat(class_to_idx)}")
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        logging.debug(f"Dataset of {len(dataset)} split into {train_size} and {test_size}")
        return train_dataset, test_dataset

    def _view_loaders(self):
        train_dataloader = self.data_state.get('train_dataloader')
        test_dataloader = self.data_state.get('test_dataloader')
        logging.debug(f"Train dataloader: {next(iter(train_dataloader))}")
        logging.debug(f"Test dataloader: {next(iter(test_dataloader))}")

    def __call__(self):
        return self.run()
