from __future__ import annotations

from dataclasses import dataclass
import logging

@dataclass
class DataConfig:
    overwrite: bool = True
    save_fig: bool = True


class DataState:
    def __init__(self):
        self._state = {}
        logging.warning(f"DataState created: {self._state}")

    def set(self, key, value):
        """Store a value in the state"""
        self._state[key] = value

    def get(self, key):
        """Retrieve a value from the state"""
        return self._state.get(key)

    def clear(self):
        """Clear all state"""
        self._state = {}