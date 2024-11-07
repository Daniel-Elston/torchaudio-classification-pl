from __future__ import annotations

from config.state_init import StateManager
from src.db.db_factory import DatabaseFactory
from utils.execution import TaskExecutor


class DatabasePipeline:
    """
    ELTL or ETL pipeline for database operations.
    """

    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe
        self.db_stage = self.state.db_config.stage

        self.db_factory = DatabaseFactory(self.state, self.db_stage)
        self.load_path, self.save_paths = self.db_factory.create_paths()
        self.steps = self.db_factory.create_steps()

    def load_fetch(self):
        self.exe._execute_steps(self.steps, stage="parent")
