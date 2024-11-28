from __future__ import annotations

from config.state_init import StateManager
from src.data.download_dataset import DownloadDataset
from src.data.label_mapping import CreateLabelMapping
from src.data.make_dataset import HDF5AudioDataset
from src.data.process_store import ProcessStoreData

# from src.data.load_dataset import LoadDataset
from src.plots.visuals import VisualiseLoader
from utils.execution import TaskExecutor


class DataPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe
        self.config = state.data_config
        self.dataset = HDF5AudioDataset(hdf5_filename=state.paths.get_path("hdf5"))

    def run(self):
        steps = [
            DownloadDataset(self.state),
            ProcessStoreData(self.state, self.dataset, self.config),
            # LoadDataset(self.state, batch_size=1),
            CreateLabelMapping(self.state, self.config),
            VisualiseLoader(self.state),
        ]
        self.exe._execute_steps(steps, stage="parent")

    def __call__(self):
        self.run()
