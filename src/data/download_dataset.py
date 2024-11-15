from __future__ import annotations

import pandas as pd
import os
from pathlib import Path
import logging
from config.state_init import StateManager
import torchaudio as ta


class DownloadDataset:
    """Load dataset and perform base processing"""
    def __init__(self, state: StateManager, view=None):
        self.state = state
        self.view = view
        self.save_path = self.state.paths.get_path('raw')

    def pipeline(self):
        self.download_data()
        if self.view:
            self._view_data()

    def download_data(self):
        directory = './data/raw/'
        if os.path.isdir(self.save_path):
            logging.debug("Data folder exists. Skipping download.")
        else:
            logging.debug(f"Downloading data to ``{directory}``.")
            trainset_speechcommands = ta.datasets.SPEECHCOMMANDS(
                f'{directory}', download=True)

    def _view_data(self):
        labels = os.listdir(self.save_path)
        logging.debug(labels)
    
    def __call__(self):
        self.pipeline()