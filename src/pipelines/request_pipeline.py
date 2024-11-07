from __future__ import annotations

from config.state_init import StateManager
from src.api.request_factory import RequestFactory
from utils.execution import TaskExecutor


class RequestPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe
        self.save_path = state.paths.get_path(f"{state.api_config.symbol}_{state.api_config.mode}")

    def main(self):
        """
        Main entry point for the pipeline.
        """
        steps = [
            (RequestFactory(self.state, self.exe), None, self.save_path),
        ]
        self.exe._execute_steps(steps, stage="parent")
