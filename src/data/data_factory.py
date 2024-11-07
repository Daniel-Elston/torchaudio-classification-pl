from __future__ import annotations

from config.state_init import StateManager
from src.data.normalise_data import (
    NormaliseCryptoHistorical,
    NormaliseCryptoLive,
    NormaliseStock,
)
from utils.execution import TaskExecutor


class DataFactory:
    """
    Factory class to create data processing objects.
    """

    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe
        self.market = state.api_config.market
        self.mode = state.api_config.mode

    def create_market_request(self):
        if self.market == "crypto" and self.mode == "live":
            return NormaliseCryptoLive().pipeline
        elif self.market == "crypto" and self.mode == "historical":
            return NormaliseCryptoHistorical().pipeline
        elif self.market == "stock":
            return NormaliseStock().pipeline

        raise ValueError(f"Invalid market or mode: {self.market}, {self.mode}.")
