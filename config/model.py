from __future__ import annotations

import logging
from pprint import pformat

import attr
import torch


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@attr.s
class ModelConfig:
    device: str = attr.ib(factory=get_device)
    subset: bool = attr.ib(default=False)
    epochs: int = attr.ib(default=2)
    batch_size: int = attr.ib(default=32)
    train_size: float = attr.ib(default=0.7)
    val_size: float = attr.ib(default=0.15)
    test_size: float = attr.ib(default=0.15)
    model: str = attr.ib(default="cnn")
    optimizer: str = attr.ib(default="adam")
    input_channels: int = attr.ib(default=1)
    hidden_channels1: int = attr.ib(default=32)
    hidden_channels2: int = attr.ib(default=64)
    learning_rate: float = attr.ib(default=0.001)
    num_classes: int = attr.ib(default=10)
    criterion: str = attr.ib(default=torch.nn.CrossEntropyLoss)

    def __attrs_post_init__(self):
        logging.info(pformat(self))


class ModelState:
    def __init__(self):
        self._state = {}

    def set(self, key, value):
        """Store a value in the state"""
        self._state[key] = value

    def get(self, key):
        """Retrieve a value from the state"""
        return self._state.get(key)

    def clear(self):
        """Clear all state"""
        self._state = {}
