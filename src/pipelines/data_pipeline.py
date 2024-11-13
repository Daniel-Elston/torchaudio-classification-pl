from __future__ import annotations

from config.state_init import StateManager
from utils.execution import TaskExecutor
from src.data.download_dataset import DownloadDataset
from src.data.process_store import ProcessStoreData


class DataPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe

    def main(self):
        steps = [
            # DownloadDataset(self.state),
            ProcessStoreData(self.state, self.exe),
        ]
        self.exe._execute_steps(steps, stage="parent")

    def __call__(self):
        self.main()