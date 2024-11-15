from __future__ import annotations

from dataclasses import dataclass, field

from config.api import ApiConfig
from config.model import ModelConfig
from config.data import DataConfig, DataState
from config.db import DatabaseConfig, DatabaseConnManager
from config.paths import PathsConfig


@dataclass
class StateManager:
    paths: PathsConfig = field(default_factory=PathsConfig)
    api_config: ApiConfig = field(default_factory=ApiConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    data_state: DataState = field(default_factory=DataState)
    db_config: DatabaseConfig = field(default_factory=DatabaseConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)

    db_manager: DatabaseConnManager = field(init=False)

    def __post_init__(self):
        self.initialize_database()
        self.validate_paths()

    def initialize_database(self):
        self.db_manager = DatabaseConnManager(self.db_config)

    def validate_paths(self):
        self.paths.validate_paths()
