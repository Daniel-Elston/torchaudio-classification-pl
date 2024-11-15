from __future__ import annotations

from config.state_init import StateManager
from utils.execution import TaskExecutor
from src.data.download_dataset import DownloadDataset
from src.data.process_store import ProcessStoreData
from src.data.load_dataset import LoadDataset
from src.plots.visuals import Visualiser
from pprint import pprint
from typing import List, Optional, Any, Dict
from src.data.create_imgs import CreateImages

class DataPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe

    def run(self):
        steps = [
            # DownloadDataset(self.state),
            # ProcessStoreData(self.state, self.exe),
            # LoadDataset(self.state, label="no", batch_size=64),
            LoadDataset(self.state, batch_size=1),
            # Visualiser(self.state),
            CreateImages(self.state),
        ]
        self.exe._execute_steps(steps, stage="parent")

    def __call__(self):
        self.run()
