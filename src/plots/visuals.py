from __future__ import annotations

import logging
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torchaudio as ta
from torch.utils.data import DataLoader

from config.state_init import StateManager
from src.models.metrics import MetricLogger


class VisualiseLoader:
    """Visualise the data loader and transforms"""

    def __init__(self, state: StateManager, dataloader: DataLoader):
        self.data_state = state.data_state
        self.data_config = state.data_config
        self.dataloader = dataloader

    def __call__(self):
        self.vis_batch()
        self.vis_waveform()
        self.vis_spectogram()
        self.vis_melspectrogram()
        self.vis_mel_freq_cepstral_coeff()

    def vis_batch(self):
        batch = next(iter(self.dataloader))
        logging.debug(pformat(batch))

    def vis_waveform(self):
        batch = next(iter(self.dataloader))

        waveform = batch["waveform"]
        sample_rate = batch["sample_rate"][0].item()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        channel = 0
        waveform_original = waveform[channel, :].view(-1)
        ax1.plot(waveform_original.numpy())
        ax1.set_title(f"Original waveform, label: {batch['label'][0]}, sample rate: {sample_rate}")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Amplitude")

        new_sample_rate = sample_rate // 10
        waveform_trans = ta.transforms.Resample(sample_rate, new_sample_rate)(
            waveform[channel, :].view(-1)
        )

        ax2.plot(waveform_trans.numpy())
        ax2.set_title(
            f"Resampled waveform, label: {batch['label'][0]}, sample rate: {new_sample_rate}"
        )
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Amplitude")

        plt.tight_layout()
        if self.data_config.save_fig:
            plt.savefig("reports/figures/waveform.png")
        plt.show()
        plt.close()

    def vis_spectogram(self):
        batch = next(iter(self.dataloader))

        waveform = batch["waveform"][0]
        sample_rate = batch["sample_rate"][0].item()
        spectrogram = ta.transforms.Spectrogram()(waveform)

        plt.figure()
        plt.title(f"Spectrogram, label: {batch['label'][0]}, sample rate: {sample_rate}")
        plt.imshow(spectrogram.log2().squeeze().numpy(), cmap="viridis")
        if self.data_config.save_fig:
            plt.savefig("reports/figures/spectrogram.png")
        plt.show()
        plt.close()

    def vis_melspectrogram(self):
        batch = next(iter(self.dataloader))

        waveform = batch["waveform"][0]
        sample_rate = batch["sample_rate"][0].item()

        mel_spectrogram = ta.transforms.MelSpectrogram(sample_rate)(waveform)

        plt.figure()
        plt.title(f"Mel Spectrogram, label: {batch['label'][0]}, sample rate: {sample_rate}")
        plt.imshow(mel_spectrogram.log2().squeeze().numpy(), cmap="viridis")
        if self.data_config.save_fig:
            plt.savefig("reports/figures/mel_spectrogram.png")
        plt.show()
        plt.close()

    def vis_mel_freq_cepstral_coeff(self):
        batch = next(iter(self.dataloader))

        waveform = batch["waveform"][0]
        sample_rate = batch["sample_rate"][0].item()

        mfcc_spectrogram = ta.transforms.MFCC(sample_rate=sample_rate)(waveform)

        plt.figure()
        plt.title(f"Mel Spectrogram, label: {batch['label'][0]}, sample rate: {sample_rate}")
        plt.imshow(mfcc_spectrogram.log2().squeeze().numpy(), cmap="viridis")
        if self.data_config.save_fig:
            plt.savefig("reports/figures/mfcc.png")
        plt.show()
        plt.close()


class VisualiseEvaluation:
    def __init__(
        self,
        state: StateManager,
        metric_logger: MetricLogger,
        idx_to_label=None,
    ):
        self.data_config = state.data_config
        self.model_config = state.model_config
        self.metric_logger = metric_logger
        self.idx_to_label = idx_to_label

    def __call__(self):
        self.print_labels()
        self.plot_loss()
        self.plot_accuracy()
        self.plot_precision_recall_f1()
        self.plot_confusion_matrix()

    def print_labels(self):
        """Labels used in the dataset"""
        unique_labels = self.metric_logger.unique_labels
        if unique_labels:
            logging.debug(f"Unique labels: {unique_labels}")
            logging.debug(f"idx_to_label: {self.idx_to_label}")
        else:
            logging.debug("Unique labels not available.")
            logging.debug(f"idx_to_label: {self.idx_to_label}")

    def plot_loss(self):
        train_losses = self.metric_logger.train_loss_list
        val_losses = self.metric_logger.val_loss_list
        train_epochs = range(1, len(train_losses) + 1)
        val_epochs = range(0, len(val_losses))
        plt.figure(figsize=(10, 6))
        plt.plot(train_epochs, train_losses, label="Train Loss")
        plt.plot(val_epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train and Validation Losses Over Epochs")
        plt.legend()
        plt.grid(True)
        if self.data_config.save_fig:
            plt.savefig(f"reports/eval/loss_e{self.model_config.epochs}.png")
        plt.show()
        plt.close()

    def plot_accuracy(self):
        train_accuracies = self.metric_logger.train_acc_list
        val_accuracies = self.metric_logger.val_acc_list
        train_epochs = range(1, len(train_accuracies) + 1)
        val_epochs = range(0, len(val_accuracies))
        plt.figure(figsize=(10, 6))
        plt.plot(train_epochs, train_accuracies, label="Train Accuracy")
        plt.plot(val_epochs, val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Train and Validation Accuracies Over Epochs")
        plt.legend()
        plt.grid(True)
        if self.data_config.save_fig:
            plt.savefig(f"reports/eval/accuracy_e{self.model_config.epochs}.png")
        plt.show()
        plt.close()

    def plot_precision_recall_f1(self):
        val_precisions = self.metric_logger.val_precision_list
        val_recalls = self.metric_logger.val_recall_list
        val_f1s = self.metric_logger.val_f1_list
        val_epochs = range(0, len(val_precisions))
        plt.figure(figsize=(10, 6))
        plt.plot(val_epochs, val_precisions, label="Validation Precision")
        plt.plot(val_epochs, val_recalls, label="Validation Recall")
        plt.plot(val_epochs, val_f1s, label="Validation F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Precision, Recall, and F1 Score Over Epochs")
        plt.legend()
        plt.grid(True)
        if self.data_config.save_fig:
            plt.savefig(f"reports/eval/precision_recall_f1_e{self.model_config.epochs}.png")
        plt.show()
        plt.close()

    def plot_confusion_matrix(self):
        confusion_matrix = self.metric_logger.confusion_matrix
        unique_labels = self.metric_logger.unique_labels
        if confusion_matrix is not None:
            cm = np.squeeze(confusion_matrix)
            mapped_labels = {
                idx: self.idx_to_label[idx] for idx in unique_labels if idx in self.idx_to_label
            }
            class_names = [mapped_labels[idx] for idx in unique_labels]

            plt.figure(figsize=(10, 7))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
            )
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title("Confusion Matrix")
            if self.data_config.save_fig:
                plt.savefig(f"reports/eval/confusion_matrix_e{self.model_config.epochs}.png")
            plt.show()
            plt.close()
        else:
            print("Confusion matrix not available.")
