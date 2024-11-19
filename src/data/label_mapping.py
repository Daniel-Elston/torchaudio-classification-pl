from __future__ import annotations

import h5py
import json
from config.state_init import StateManager
from config.data import DataConfig


def create_label_mapping(hdf5_filename, save_path=None):
    with h5py.File(hdf5_filename, 'r') as h5f:
        labels = list(h5f.keys())
        labels = sorted(labels)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(label_to_idx, f)

    return label_to_idx, idx_to_label

def load_label_mapping(load_path):
    with open(load_path, 'r') as f:
        label_to_idx = json.load(f)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


class CreateLabelMapping:
    def __init__(self, state: StateManager, config: DataConfig):
        self.state = state
        self.data_state = state.data_state
        self.config = config
        self.hdf5_filename = 'data/audio_data.hdf5'
        self.save_path = 'reports/label_mapping.json'
    
    def __call__(self):
        label_to_idx, idx_to_label = create_label_mapping(
            self.hdf5_filename, save_path=self.save_path
        )
        self.data_state.set('label_to_idx', label_to_idx)
        self.data_state.set('idx_to_label', idx_to_label)
    
    @staticmethod
    def create_label_mapping(hdf5_filename, save_path=None):
        with h5py.File(hdf5_filename, 'r') as h5f:
            labels = list(h5f.keys())
            labels = sorted(labels)
            label_to_idx = {label: idx for idx, label in enumerate(labels)}
            idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(label_to_idx, f)
        return label_to_idx, idx_to_label

    @staticmethod
    def load_label_mapping(load_path):
        with open(load_path, 'r') as f:
            label_to_idx = json.load(f)
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        return label_to_idx, idx_to_label
