from __future__ import annotations

from config.api import CryptoConfig, StockConfig
from config.state_init import StateManager
from src.api.request_crypto import RequestHistoricalCrypto, RequestLiveCrypto
from src.api.request_stock import RequestHistoricalStock, RequestLiveStock
from utils.execution import TaskExecutor


class RequestFactory:
    """
    Factory class to create market request objects.
    """

    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe
        self.base_config = state.api_config

    def class_factory(self):
        class_map = {
            ("crypto", "live"): RequestLiveCrypto,
            ("crypto", "historical"): RequestHistoricalCrypto,
            ("stock", "live"): RequestLiveStock,
            ("stock", "historical"): RequestHistoricalStock,
        }
        return class_map.get((self.base_config.market, self.base_config.mode))

    def config_factory(self):
        config_map = {"crypto": CryptoConfig(), "stock": StockConfig()}
        return config_map.get(self.base_config.market)

    def create_market_request(self):
        request_class = self.class_factory()
        request_config = self.config_factory()
        if request_class and request_config:
            return request_class(self.state, request_config).pipeline
        raise ValueError(
            f"Invalid market or mode: {self.base_config.market}, {self.base_config.mode}."
        )

    def __call__(self):
        return self.create_market_request()
