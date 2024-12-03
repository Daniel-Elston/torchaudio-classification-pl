from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from config.model import ModelConfig


class CNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super(CNN, self).__init__()
        self.config = config

        # Input will be [batch_size, 1, 80, 101]
        self.conv1 = nn.Conv2d(1, config.hidden_channels1, kernel_size=5)
        self.conv2 = nn.Conv2d(config.hidden_channels1, config.hidden_channels2, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()

        # Compute the input size for the first fully connected layer
        self.fc_input_size = self.compute_fc_size()

        self.fc1 = nn.Linear(self.fc_input_size, 50)
        self.fc2 = nn.Linear(50, config.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def compute_fc_size(self):
        # Create a dummy input matching the spectrogram dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 80, 101)  # Match spectrogram dimensions
            x = F.relu(F.max_pool2d(self.conv1(dummy_input), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = self.flatten(x)
            fc_input_size = x.shape[1]
        return fc_input_size
