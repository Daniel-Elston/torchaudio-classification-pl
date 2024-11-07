from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pprint import pformat


@dataclass
class ApiConfig:
    """Base configuration class for market requests (e.g., crypto or stock)."""

    symbol: str = "NVDA"
    market: str = "stock"  # 'crypto' or 'stock'
    mode: str = "live"  # "live" or "historical"
    sleep_interval: int = 60

    def __post_init__(self):
        logging.debug(f"Initialized base market config: {pformat(self.__dict__)}")


@dataclass
class CryptoConfig(ApiConfig):
    """Configuration for crypto market requests."""

    currency: str = "USDT"
    exchange_name: str = "binance"
    interval: str = "15m"
    batch_size: int = 2
    max_items: int = 6
    since: str = "31/10/2024"
    limit: int = 1000

    def __post_init__(self):
        super().__post_init__()
        if self.mode == "historical":
            self.interval = "30m"
        logging.debug(f"Initialized crypto config: {pformat(self.__dict__)}")


@dataclass
class StockConfig(ApiConfig):
    """Configuration for stock market requests."""

    base_url: str = "https://www.alphavantage.co/query"
    interval: str = "15min"
    outputsize: str = "compact"
    apikey: str = os.getenv("ALPHA_VANTAGE_API")
    function: str = None

    def __post_init__(self):
        super().__post_init__()
        self.apikey = os.getenv("ALPHA_VANTAGE_API")
        self.function = "TIME_SERIES_INTRADAY" if self.mode == "live" else "TIME_SERIES_DAILY"
        if self.mode == "live":
            self.interval = "1min"
        logging.debug(f"Initialized stock config: {pformat(self.__dict__)}")
