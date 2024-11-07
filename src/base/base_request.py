from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import aiohttp

from config.api import CryptoConfig, StockConfig
from config.state_init import StateManager
from utils.execution import TaskExecutor
from utils.file_access import save_json


class BaseMarketRequest(ABC):
    """
    Base class for all market requests
    """

    def __init__(self, state: StateManager, params: Union[CryptoConfig, StockConfig]):
        self.params = params
        self.save_path = state.paths.get_path(f"{state.api_config.symbol}_{state.api_config.mode}")

    @abstractmethod
    async def fetch_data(self):
        """Abstract method to fetch data"""
        pass

    def pipeline(self, df):
        steps = [
            self.run_data_fetch,
        ]
        for step in steps:
            df = TaskExecutor.run_child_step(step, df)
        return df

    def run_data_fetch(self, _):
        asyncio.run(self.fetch_data())


class BaseCryptoRequest(BaseMarketRequest):
    """
    Base class for all crypto requests
    """

    def __init__(self, state, params: CryptoConfig):
        super().__init__(state, params)

    async def batch_save_helper(self, batch, path: Path):
        """Helper method to save batch data conditionally."""
        logging.debug(f"Saving batch of size {len(batch)} to: {path}")
        await save_json(batch, path)
        batch.clear()


class BaseStockRequest(BaseMarketRequest):
    """
    Base class for all stock requests
    """

    def __init__(self, state, params: StockConfig):
        super().__init__(state, params)

    async def perform_request(self):
        """
        Shared request logic for stock data.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(self.params.base_url, params=self.params.__dict__) as response:
                if response.status == 200:
                    data = await response.json()
                    await save_json(data, self.save_path)
                else:
                    logging.error(f"Failed fetch: {response.status}, {await response.text()}")
