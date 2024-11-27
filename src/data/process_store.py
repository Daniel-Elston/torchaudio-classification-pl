from __future__ import annotations

from pathlib import Path
import logging
from config.state_init import StateManager
from utils.execution import TaskExecutor
from config.data import DataConfig

import h5py
import torchaudio as ta
import torch


class ProcessStoreData:
    def __init__(self, state: StateManager, exe: TaskExecutor, config: DataConfig):
        self.state = state
        self.exe = exe
        self.labels = config.labels
        self.load_path = state.paths.get_path('raw')
        self.hdf5_path = state.paths.get_path('hdf5')

    def pipeline(self):
        with h5py.File(self.hdf5_path, 'w') as h5f:
            for label in self.labels:
                if label == '_background_noise_':
                    pass
                else:
                    logging.debug(f"Processing label: {label}")
                    self.process_and_save_label(label, h5f)
        self.view_hdf5(self.hdf5_path)


    def process_and_save_label(self, label, h5f):
        """Process a single label and save its data to the HDF5 file"""
        label_group = h5f.create_group(label)
        files = sorted(Path(self.load_path, label).glob('*.wav'))
        for file_path in files:
            waveform, sample_rate = ta.load(str(file_path))
            waveform, sample_rate = waveform.to(torch.float32), int(sample_rate/10)
            speaker_id = file_path.stem.split('_')[0]
            utterance_number = int(file_path.stem.split('_')[-1])
            
            ds_name = f"{speaker_id}_{utterance_number}"
            ds = label_group.create_dataset(
                name=ds_name,
                data=waveform.numpy(),
                compression="gzip"
            )
            ds.attrs['sample_rate'] = sample_rate
            ds.attrs['speaker_id'] = speaker_id
            ds.attrs['utterance_number'] = utterance_number

    def view_hdf5(self, hdf5_filename):
        """View the structure and contents of an HDF5 file"""
        with h5py.File(hdf5_filename, 'r') as h5f:
            for label in h5f.keys():
                logging.debug(f"Label: {label}")
                for ds_name in h5f[label].keys():
                    ds = h5f[label][ds_name]
                    logging.debug(
                        f" - {ds_name, type(ds_name)} | Sample Rate: {ds.attrs['sample_rate'], type(ds.attrs['sample_rate'])}, "
                        f"Speaker ID: {ds.attrs['speaker_id'], type(ds.attrs['speaker_id'])}, "
                        f"Utterance: {ds.attrs['utterance_number'], type(ds.attrs['utterance_number'])}")

    def __call__(self):
        self.pipeline()