from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import ccxt.pro as ccxtpro

from config.api import CryptoConfig
from config.state_init import StateManager
from src.base.base_request import BaseCryptoRequest
from utils.file_access import temp_file_reset


class RequestLiveCrypto(BaseCryptoRequest):
    def __init__(self, state: StateManager, params: CryptoConfig):
        super().__init__(state, params)

    async def fetch_data(self):
        await temp_file_reset(self.save_path)
        batch = []
        async with getattr(ccxtpro, self.params.exchange_name)() as exchange:
            while len(batch) < self.params.batch_size:
                ticker_symbol = f"{self.params.symbol}/{self.params.currency}"
                ticker = await exchange.fetch_ticker(ticker_symbol)
                batch.append(ticker)
            await self.batch_save_helper(batch, self.save_path)
            await asyncio.sleep(1)


class RequestHistoricalCrypto(BaseCryptoRequest):
    def __init__(self, state: StateManager, params: CryptoConfig):
        super().__init__(state, params)

    async def fetch_data(self):
        exchange = getattr(ccxtpro, self.params.exchange_name)()
        await temp_file_reset(self.save_path)
        try:
            since_timestamp = int(
                datetime.strptime(self.params.since, "%d/%m/%Y").timestamp() * 1000
            )
            batch = []
            while True:
                logging.debug(
                    f"Fetching historical data for {self.params.symbol}/{self.params.currency} starting from {self.params.since}"
                )
                ohlcv = await exchange.fetch_ohlcv(
                    f"{self.params.symbol}/{self.params.currency}",
                    self.params.interval,
                    since=since_timestamp,
                    limit=self.params.limit,
                )
                if not ohlcv:
                    break
                batch.extend(ohlcv)
                since_timestamp = ohlcv[-1][0] + 60000  # Move time forward to fetch next batch
                await self.batch_save_helper(batch, self.save_path)
        finally:
            await exchange.close()
