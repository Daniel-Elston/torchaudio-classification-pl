from __future__ import annotations

import torchaudio as ta
import matplotlib.pyplot as plt
import numpy as np
from config.state_init import StateManager


class VisualiseLoader:
    """Load dataset and perform base processing"""
    def __init__(self, state: StateManager):
        self.data_state = state.data_state
        
    def pipeline(self):
        self.vis_waveform()
        self.vis_spectogram()
        self.vis_melspectrogram()
        self.vis_mel_freq_cepstral_coeff()

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
        plt.close()
    
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
        plt.close()

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
        plt.close()
        
    def vis_mel_freq_cepstral_coeff(self):
        self.dataloader = self.data_state.get('dataloader')
        batch = next(iter(self.dataloader))
        
        waveform = batch['waveform'][0]
        sample_rate = batch['sample_rate'][0].item()
        
        mfcc_spectrogram = ta.transforms.MFCC(sample_rate= sample_rate)(waveform)

        plt.figure()
        plt.title(
            f"Mel Spectrogram, label: {batch['label'][0]}, sample rate: {sample_rate}")
        plt.imshow(mfcc_spectrogram.log2().squeeze().numpy(), cmap='viridis')
        plt.show()
        plt.close()
        
    def __call__(self):
        return self.pipeline()



class VisualiseEvaluation:
    def __init__(
        self,
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        val_precisions,
        val_recalls,
        val_f1s,
        confusion_matrix=None
    ):
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_accuracies = train_accuracies
        self.val_accuracies = val_accuracies
        self.val_precisions = val_precisions
        self.val_recalls = val_recalls
        self.val_f1s = val_f1s
        self.confusion_matrix = confusion_matrix

    def pipeline(self):
        self.plot_loss()
        self.plot_accuracy()
        self.plot_precision_recall_f1()
        self.plot_confusion_matrix()
        # If you have confusion matrix data, you can plot it here as well
        # self.plot_confusion_matrix()

    def plot_loss(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Losses Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_accuracy(self):
        epochs = range(1, len(self.train_accuracies) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_accuracies, label='Train Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train and Validation Accuracies Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_precision_recall_f1(self):
        epochs = range(1, len(self.val_precisions) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.val_precisions, label='Validation Precision')
        plt.plot(epochs, self.val_recalls, label='Validation Recall')
        plt.plot(epochs, self.val_f1s, label='Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Precision, Recall, and F1 Score Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_confusion_matrix(self):
        if self.confusion_matrix is not None:
            plt.figure(figsize=(10, 7))
            sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title("Confusion Matrix")
            plt.show()
        else:
            print("Confusion matrix not available.")
    

    def __call__(self):
        self.pipeline()