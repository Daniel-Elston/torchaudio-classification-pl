from __future__ import annotations

from config.state_init import StateManager
from utils.execution import TaskExecutor
from src.data.download_dataset import DownloadDataset
from src.data.process_store import ProcessStoreData
from src.data.load_dataset import LoadDataset
from src.plots.visuals import Visualiser
from src.data.create_imgs import CreateImages
from src.data.load_imgs import LoadImages


class DataPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe
        self.config = state.data_config

    def run(self):
        steps = [
            DownloadDataset(self.state),
            ProcessStoreData(self.state, self.exe, self.config),
            LoadDataset(self.state, batch_size=1),
            Visualiser(self.state),
            CreateImages(self.state),
            LoadImages(self.state, self.config, img_type='spectograms', batch_size=16),
        ]
        self.exe._execute_steps(steps, stage="parent")

    def __call__(self):
        self.run()
