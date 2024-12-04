from __future__ import annotations

from config.state_init import StateManager
from src.base.base_trainer import BaseTrainer
from src.data.data_module import HDF5DataModule
from src.data.label_mapping import CreateLabelMapping
from src.data.make_dataset import HDF5AudioDataset
from src.models.cnn import CNN
from src.models.metrics import MetricLogger
from src.models.train import ModelTrainer
from src.plots.visuals import VisualiseEvaluation
from utils.execution import TaskExecutor
from utils.file_access import load_label_mapping, reverse_mapping


class ModelPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe
        self.config = state.model_config
        self.hdf5_path = state.paths.get_path("hdf5")
        self.label_to_idx = load_label_mapping()
        self.idx_to_label = reverse_mapping(load_label_mapping())

        self.dataset = HDF5AudioDataset(
            hdf5_filename=self.hdf5_path,
            transform=None,
            label_to_idx=self.label_to_idx,
            idx_to_label=self.idx_to_label,
        )
        self.data_module = HDF5DataModule(
            config=self.config,
            dataset=self.dataset,
            hdf5_filename=self.hdf5_path,
            subset=self.config.subset,
        )
        self.model = CNN(self.config)
        self.trainer_model = BaseTrainer(
            self.model,
            self.config,
        )
        self.metric_logger = MetricLogger(self.config.num_classes, self.idx_to_label)

    def __call__(self):
        self.run()

    def run(self):
        steps = [
            CreateLabelMapping(self.state, self.config),
            ModelTrainer(
                self.state,
                self.config,
                self.data_module,
                self.model,
                self.trainer_model,
                self.metric_logger,
            ),
            VisualiseEvaluation(
                self.state,
                self.metric_logger,
                self.idx_to_label,
            ),
        ]
        self.exe._execute_steps(steps, stage="parent")
