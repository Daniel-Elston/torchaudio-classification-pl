from __future__ import annotations

import logging
from pprint import pformat
import attr


@attr.s
class ModelConfig:
    save_fig: bool = attr.ib(default=True)
