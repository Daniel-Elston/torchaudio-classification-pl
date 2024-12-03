from __future__ import annotations

import attr


def default_labels():
    return ["cat", "dog", "go", "happy", "left", "no", "off", "on", "yes", "zero"]


@attr.s
class DataConfig:
    overwrite: bool = attr.ib(default=True)
    save_fig: bool = attr.ib(default=False)
    labels: list = attr.ib(factory=default_labels)
    train_size: float = attr.ib(default=0.5)
    subset: bool = attr.ib(default=True)
    batch_size: int = attr.ib(default=32)
    shuffle: bool = attr.ib(default=True)
    num_workers: int = attr.ib(default=4)


class DataState:
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
