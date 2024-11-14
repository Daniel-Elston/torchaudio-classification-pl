from __future__ import annotations

import logging
from pprint import pprint, pformat
from typing import List, Optional, Any, Dict

from torch.utils.data import DataLoader
from src.data.make_dataset import HDF5AudioDataset
from src.data.make_dataset import LabelSampler
from config.state_init import StateManager


class LoadDataset:
    def __init__(self, state: StateManager, label=None, batch_size=64, view=False):
        self.data_state = state.data_state
        self.label = label
        self.batch_size = batch_size
        self.view = view
        self.hdf5_filename = 'audio_data.hdf5'
        self.dataloader = None

    def run(self):
        dataset = HDF5AudioDataset(hdf5_filename=self.hdf5_filename)
        if self.label:
            sampler = LabelSampler(dataset, label=self.label)
            self.dataloader = DataLoader(
                dataset, sampler=sampler, batch_size=self.batch_size, num_workers=4)
        else:
            self.dataloader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        if self.view:
            self._debug_batch()
            
        self.data_state.set('dataloader', self.dataloader)
        logging.info(
            f"Dataset loader created: {self.dataloader.__class__.__name__} and stored in {self.data_state.__class__.__name__}")
    
    def _debug_batch(self):
        for batch in self.dataloader:
            if self.view:
                logging.debug(f"Sample batch: {pformat(batch)}")
                for i, waveform in enumerate(batch['waveform']):
                    logging.debug(f"Sample {i+1} waveform size, dtype: {waveform.size()}, {waveform.dtype}")
                    break
                break

    def __call__(self):
        # if self.dataloader is None:
        #     self.data_state.set('dataloader', self.dataloader)
        # else:
        #     logging.error("Dataloader creation failed in LoadDataset")
        return self.run()
