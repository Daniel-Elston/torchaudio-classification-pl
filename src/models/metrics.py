from __future__ import annotations

import pytorch_lightning as pl
import torch
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class MetricLogger(pl.callbacks.Callback):
    def __init__(self, num_classes, idx_to_label=None):
        super().__init__()
        self.num_classes = num_classes
        self.idx_to_label = idx_to_label

        # Initialize metrics
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        # Lists to store metrics
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []
        self.val_precision_list = []
        self.val_recall_list = []
        self.val_f1_list = []

        # For confusion matrix
        self.val_preds = []
        self.val_labels = []
        self.confusion_matrix = None

        self.unique_labels = set()

    def on_train_start(self, trainer, pl_module):
        """Log class labels at the start of training"""
        if self.idx_to_label is not None:
            labels_text = ", ".join(
                [f"{idx}: {label}" for idx, label in sorted(self.idx_to_label.items())]
            )
            trainer.logger.experiment.add_text("Class Labels", labels_text, global_step=0)

    def on_train_epoch_start(self, trainer, pl_module):
        """Reset train metrics at the start of each epoch"""
        self.train_accuracy.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update train metrics"""
        preds = torch.argmax(outputs["logits"], dim=1)
        labels = batch["label"]
        self.train_accuracy.update(preds, labels)

    def on_train_epoch_end(self, trainer, pl_module):
        """Compute and store train metrics"""
        train_acc = self.train_accuracy.compute()
        self.train_acc_list.append(train_acc.cpu().item())

        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.train_loss_list.append(train_loss.cpu().item())

    def on_validation_epoch_start(self, trainer, pl_module):
        """Reset validation metrics at the start of each epoch"""
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_preds = []
        self.val_labels = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Update validation metrics"""
        preds = torch.argmax(outputs["logits"], dim=1)
        labels = batch["label"]

        self.val_accuracy.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_f1.update(preds, labels)

        self.val_preds.append(preds.cpu())
        self.val_labels.append(labels.cpu())

        self.unique_labels.update(labels.cpu().numpy().tolist())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and store validation metrics"""
        val_acc = self.val_accuracy.compute()
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()
        val_f1 = self.val_f1.compute()

        self.val_acc_list.append(val_acc.cpu().item())
        self.val_precision_list.append(val_precision.cpu().item())
        self.val_recall_list.append(val_recall.cpu().item())
        self.val_f1_list.append(val_f1.cpu().item())

        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.val_loss_list.append(val_loss.cpu().item())

        # Compute confusion matrix with specified labels
        preds = torch.cat(self.val_preds)
        labels = torch.cat(self.val_labels)
        self.confusion_matrix = confusion_matrix(
            labels.numpy(), preds.numpy(), labels=sorted(self.unique_labels)
        )

        # Log metrics
        trainer.logger.experiment.add_scalar("Validation/Accuracy", val_acc, trainer.current_epoch)
        trainer.logger.experiment.add_scalar(
            "Validation/Precision", val_precision, trainer.current_epoch
        )
        trainer.logger.experiment.add_scalar(
            "Validation/Recall", val_recall, trainer.current_epoch
        )
        trainer.logger.experiment.add_scalar("Validation/F1_Score", val_f1, trainer.current_epoch)
