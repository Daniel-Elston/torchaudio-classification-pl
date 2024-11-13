from __future__ import annotations

import logging
from pprint import pprint, pformat

from torch.utils.data import DataLoader
from src.data.make_dataset import HDF5AudioDataset
from src.data.make_dataset import LabelSampler


class LoadDataset:
    def __init__(self, label=None, batch_size=64, view=False):
        self.label = label
        self.batch_size = batch_size
        self.view = view
        self.hdf5_filename = 'audio_data.hdf5'

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
            for batch in self.dataloader:
                logging.debug(f"Sample batch: {pformat(batch)}")
                for i, waveform in enumerate(batch['waveform']):
                    logging.debug(f"Sample {i+1} waveform size, dtype: {waveform.size()}, {waveform.dtype}")
                    break
                break

    def __call__(self):
        self.run()