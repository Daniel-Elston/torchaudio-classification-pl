from __future__ import annotations

from torch import nn
from torch.nn import functional as F
from config.model import ModelConfig


class CNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.conv1 = nn.Conv2d(config.input_channels, config.hidden_channels1, kernel_size=5)
        self.conv2 = nn.Conv2d(config.hidden_channels1, config.hidden_channels2, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(51136, 50)
        self.fc2 = nn.Linear(50, config.num_classes)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)