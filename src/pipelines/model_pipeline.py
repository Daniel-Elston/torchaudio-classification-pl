from __future__ import annotations

from config.state_init import StateManager
from utils.execution import TaskExecutor


class ModelPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe
        self.config = state.model_config

    def run(self):
        steps = [
            
        ]
        self.exe._execute_steps(steps, stage="parent")

    def __call__(self):
        self.run()
