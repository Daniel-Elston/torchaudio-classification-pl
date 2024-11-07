from __future__ import annotations

from config.api import StockConfig
from config.state_init import StateManager
from src.base.base_request import BaseStockRequest


class RequestLiveStock(BaseStockRequest):
    def __init__(self, state: StateManager, params: StockConfig):
        super().__init__(state, params)

    async def fetch_data(self):
        await self.perform_request()


class RequestHistoricalStock(BaseStockRequest):
    def __init__(self, state: StateManager, params: StockConfig):
        super().__init__(state, params)

    async def fetch_data(self):
        await self.perform_request()
