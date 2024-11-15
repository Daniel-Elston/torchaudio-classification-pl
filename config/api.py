from __future__ import annotations

import logging
from dataclasses import dataclass
from pprint import pformat


@dataclass
class ApiConfig:
    sleep_interval: int = 60

    def __post_init__(self):
        logging.debug(f"Initialized base market config: {pformat(self.__dict__)}")
