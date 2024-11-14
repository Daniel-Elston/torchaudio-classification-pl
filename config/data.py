from __future__ import annotations

from dataclasses import dataclass
import attr
import logging
from pathlib import Path


def default_labels():
    return ['cat', 'dog', 'go', 'happy', 'left', 'no', 'off', 'on', 'yes', 'zero']
    # self.labels = sorted([p.name for p in Path(load_path).iterdir() if p.is_dir()])

@attr.s
class DataConfig:
    overwrite: bool = attr.ib(default=True)
    save_fig: bool = attr.ib(default=True)
    labels: list = attr.ib(factory=default_labels)


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