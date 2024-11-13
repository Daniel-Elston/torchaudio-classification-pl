from __future__ import annotations

import torch
import h5py
from torch.utils.data import Dataset, DataLoader, Sampler


class HDF5AudioDataset(Dataset):
    """
    HDF5Dataset
    """
    def __init__(self, hdf5_filename):
        self.hdf5_filename = hdf5_filename
        self.labels = None
        self.dataset_index = None
        self._init_index()

    def _init_index(self):
        """Open the file temporarily to read the index"""
        with h5py.File(self.hdf5_filename, 'r') as h5f:
            self.labels = list(h5f.keys())
            self.dataset_index = [
                (label, key)
                for label in self.labels
                for key in h5f[label].keys()
            ]

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, idx):
        """Open the HDF5 file for each access"""
        with h5py.File(self.hdf5_filename, 'r') as h5f:
            label, key = self.dataset_index[idx]
            data = h5f[label][key]
            waveform = torch.from_numpy(data[()])
            sample_rate = data.attrs['sample_rate']
            speaker_id = data.attrs['speaker_id']
            utterance_number = data.attrs['utterance_number']
            
            # if not size 16000: remove from the dataset
            if waveform.size()[1] != 16000:
                del self.dataset_index[idx]
                return self.__getitem__(idx)

        return {
            'label': label,
            'waveform': waveform,
            'sample_rate': sample_rate,
            'speaker_id': speaker_id,
            'utterance_number': utterance_number
        }


class LabelSampler(Sampler):
    def __init__(self, dataset, label):
        self.indices = [i for i, (lbl, _) in enumerate(dataset.dataset_index) if lbl == label]

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)
    
    def __repr__(self):
        return f"LabelSampler(...{self.indices[-3:]})"