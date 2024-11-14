from __future__ import annotations

import logging
from config.state_init import StateManager
import matplotlib.pyplot as plt
import torchaudio as ta
import os
from pathlib import Path


class CreateImages:
    """Creates spectrogram and MFCC images"""
    def __init__(self, state: StateManager):
        self.data_state = state.data_state
        self.save_path = state.paths.get_path('processed')

    def run(self):
        dataloader = self.data_state.get('dataloader')
        
        for batch_num, batch in enumerate(dataloader):
            # Retrieve waveform, label, and metadata
            waveform, label, sample_rate, speaker_id, utterance_number = self.process_batch(batch)
            logging.debug(f"Processing batch {batch_num}, label: {label}, speaker id: {speaker_id}")
            
            # Ensure directory for the label exists
            self._check_data_dir(label)

            # Generate and save spectrogram image
            self.generate_and_save_spectrogram(waveform, label, batch_num)
            
            # Generate and save MFCC spectrogram image
            self.generate_and_save_mfcc(waveform, sample_rate, label, batch_num)

    def process_batch(self, batch):
        """Extract waveform, label, sample rate, speaker ID, and utterance number from batch"""
        waveform = batch['waveform'][0]
        label = batch['label'][0]
        sample_rate = batch['sample_rate'][0].item()
        speaker_id = batch['speaker_id'][0]
        utterance_number = batch['utterance_number'][0].item()
        return waveform, label, sample_rate, speaker_id, utterance_number

    def generate_and_save_spectrogram(self, waveform, label, batch_num):
        """Generate and save a spectrogram image for the given waveform"""
        spectrogram_tensor = ta.transforms.Spectrogram()(waveform)
        file_path = Path(f'{self.save_path}/spectrograms/{label}/spec_img_{batch_num}.png')
        
        plt.figure()
        plt.imsave(file_path, spectrogram_tensor[0].log2().squeeze().numpy(), cmap='viridis')
        plt.close()

    def generate_and_save_mfcc(self, waveform, sample_rate, label, batch_num):
        """Generate and save an MFCC spectrogram image for the given waveform"""
        mfcc_spectrogram = ta.transforms.MFCC(sample_rate=sample_rate)(waveform)
        file_path = Path(f'{self.save_path}/mfcc_spectrograms/{label}/spec_img_{batch_num}.png')
        
        plt.figure()
        plt.imshow(mfcc_spectrogram[0].log2().squeeze().numpy(), cmap='viridis')
        plt.colorbar(label="Log-MFCC Coefficients")
        plt.savefig(file_path, dpi=100)
        plt.close()

    def _check_data_dir(self, label):
        """Check if the directory for the given label exists, and create it if not"""
        mfcc_directory = f"{self.save_path}/mfcc_spectrograms/{label}"
        if not os.path.isdir(mfcc_directory):
            os.makedirs(mfcc_directory)
        spec_directory = f"{self.save_path}/spectrograms/{label}"
        if not os.path.isdir(spec_directory):
            os.makedirs(spec_directory)


    def __call__(self):
        return self.run()
