from __future__ import annotations

from config.state_init import StateManager
from src.data.data_factory import DataFactory
from utils.execution import TaskExecutor


class DataPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe
        base_path = f"{state.api_config.symbol}_{state.api_config.mode}"
        self.load_path = state.paths.get_path(base_path)
        self.transform_path = state.paths.get_path(f"{base_path}_transform")

    def main(self):
        steps = [
            (
                DataFactory(self.state, self.exe).create_market_request(),
                self.load_path,
                self.transform_path,
            ),
        ]
        self.exe._execute_steps(steps, stage="parent")
