from __future__ import annotations

from torch.utils.data import DataLoader

from config.state_init import StateManager
from src.data.download_dataset import DownloadDataset
from src.data.make_dataset import HDF5AudioDataset
from src.data.process_store import ProcessStoreData
from src.plots.visuals import VisualiseLoader
from utils.execution import TaskExecutor
from utils.file_access import load_label_mapping, reverse_mapping


class DataPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe
        self.config = state.data_config
        self.hdf5_path = state.paths.get_path("hdf5")
        self.label_to_idx = load_label_mapping()
        self.idx_to_label = reverse_mapping(load_label_mapping())

        self.dataset = HDF5AudioDataset(
            hdf5_filename=self.hdf5_path,
            transform=None,
            label_to_idx=self.label_to_idx,
            idx_to_label=self.idx_to_label,
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
        )

    def run(self):
        steps = [
            DownloadDataset(self.state),
            ProcessStoreData(self.state, self.config),
            VisualiseLoader(self.state, self.dataloader),
        ]
        self.exe._execute_steps(steps, stage="parent")

    def __call__(self):
        self.run()
