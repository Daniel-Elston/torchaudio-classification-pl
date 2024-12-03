from __future__ import annotations

import logging

import h5py
import torch
import torch.nn.functional as F
import torchaudio as ta
import torchaudio.transforms as T
from torch.utils.data import Dataset


class HDF5AudioDataset(Dataset):
    def __init__(self, hdf5_filename, transform=None, label_to_idx=None, idx_to_label=None):
        self.hdf5_filename = hdf5_filename
        self.transform = transform
        self.dataset_index = []
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        self._init_index()
        self.spectrogram_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_mels=80,
            n_fft=400,
            hop_length=160,
        )

    def _init_index(self):
        with h5py.File(self.hdf5_filename, "r") as h5f:
            self.labels = list(h5f.keys())
            for label in self.labels:
                for key in h5f[label].keys():
                    data = h5f[label][key]
                    waveform = torch.from_numpy(data[()])
                    if waveform.size(-1) == 16000:
                        self.dataset_index.append((label, key))

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, idx):
        label_str, key = self.dataset_index[idx]

        if self.label_to_idx:
            label = self.label_to_idx[label_str]

        with h5py.File(self.hdf5_filename, "r") as h5f:
            data = h5f[label_str][key]
            waveform = torch.from_numpy(data[()])
            sample_rate = data.attrs["sample_rate"]

        spectrogram = self.spectrogram_transform(waveform)

        spectrogram = ta.functional.amplitude_to_DB(
            spectrogram, multiplier=10, amin=1e-10, db_multiplier=0
        )

        logging.debug(f"Shape before adding dimensions: {spectrogram.shape}")
        spectrogram = spectrogram.unsqueeze(0)  # .unsqueeze(0)  # Shape: [1, 1, 80, 101]
        logging.debug(f"Shape after adding dimensions: {spectrogram.shape}")
        spectrogram = F.interpolate(
            spectrogram, size=(80, 101), mode="bilinear", align_corners=False
        )
        logging.debug(f"Shape after interpolation: {spectrogram.shape}")
        spectrogram = spectrogram.squeeze(0)  # Shape: [1, 80, 101]
        logging.debug(f"Shape after removing dimensions: {spectrogram.shape}")

        if self.transform:
            spectrogram = self.transform(spectrogram)

        label = torch.tensor(label, dtype=torch.long)
        return {
            "label": label,
            "spectrogram": spectrogram,
            "waveform": waveform,
            "sample_rate": sample_rate,
        }
