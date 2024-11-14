from __future__ import annotations

import torchaudio as ta
import matplotlib.pyplot as plt
import numpy as np
from config.state_init import StateManager

from typing import List, Optional, Any, Dict
from src.data.load_dataset import LoadDataset
from config.data import DataState
import logging
from pprint import pprint


class Visualiser:
    """Load dataset and perform base processing"""
    def __init__(self, state: StateManager):
        self.data_state = state.data_state

    def vis_waveform(self):
        self.dataloader = self.data_state.get('dataloader')
        batch = next(iter(self.dataloader))
        
        waveform = batch['waveform']
        sample_rate = batch['sample_rate'][0].item()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        channel = 0
        waveform_original = waveform[channel, :].view(-1)
        ax1.plot(waveform_original.numpy())
        ax1.set_title(
            f"Original waveform, label: {batch['label'][0]}, sample rate: {sample_rate}")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Amplitude")

        new_sample_rate = sample_rate // 10
        waveform_trans = ta.transforms.Resample(
            sample_rate,
            new_sample_rate
        )(waveform[channel, :].view(-1))
        
        ax2.plot(waveform_trans.numpy())
        ax2.set_title(
            f"Resampled waveform, label: {batch['label'][0]}, sample rate: {new_sample_rate}")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Amplitude")

        plt.tight_layout()
        plt.show()
    
    def vis_spectogram(self):
        self.dataloader = self.data_state.get('dataloader')
        batch = next(iter(self.dataloader))
        
        waveform = batch['waveform'][0]
        sample_rate = batch['sample_rate'][0].item()
        spectrogram = ta.transforms.Spectrogram()(waveform)
        
        plt.figure()
        plt.title(
            f"Spectrogram, label: {batch['label'][0]}, sample rate: {sample_rate}")
        plt.imshow(spectrogram.log2().squeeze().numpy(), cmap='viridis')
        plt.show()

    def vis_melspectrogram(self):
        self.dataloader = self.data_state.get('dataloader')
        batch = next(iter(self.dataloader))
        
        waveform = batch['waveform'][0]
        sample_rate = batch['sample_rate'][0].item()
        
        mel_spectrogram = ta.transforms.MelSpectrogram(sample_rate)(waveform)

        plt.figure()
        plt.title(
            f"Mel Spectrogram, label: {batch['label'][0]}, sample rate: {sample_rate}")
        plt.imshow(mel_spectrogram.log2().squeeze().numpy(), cmap='viridis')
        plt.show()
        
    def __call__(self):
        self.vis_waveform()
        self.vis_spectogram()
        self.vis_melspectrogram()