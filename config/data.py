from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataConfig:
    overwrite: bool = True
    save_fig: bool = True
