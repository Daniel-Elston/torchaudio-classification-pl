from __future__ import annotations

from config.state_init import StateManager
from utils.execution import TaskExecutor

from src.data.data_module import HDF5DataModule
from src.models.init_model import InitialiseModel
from src.models.train import ModelTrainer
from src.models.validate import ValidateModel
from src.models.save_model import SaveModel
from src.models.load_model import LoadModel
from src.models.evaluate import EvaluateModel
from src.plots.visuals import VisualiseEvaluation
from src.data.label_mapping import CreateLabelMapping
from utils.file_access import load_label_mapping
from src.models.cnn import CNN
from src.base.base_trainer import BaseTrainer


class ModelPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe
        self.config = state.model_config
        self.hdf5_path = state.paths.get_path('hdf5')
        self.data_module = HDF5DataModule(
            config=self.config,
            hdf5_filename=self.hdf5_path,
            subset=self.config.subset,
            label_to_idx=load_label_mapping(),
        )
        self.model = CNN(
            self.config
        )
        self.trainer_model = BaseTrainer(
            self.model, 
            self.config, 
        )
        
    def run(self):
        steps = [
            CreateLabelMapping(
                self.state,
                self.config
            ),
            ModelTrainer(
                self.state,
                self.config,
                self.data_module,
                self.model,
                self.trainer_model
            ),
            # ValidateModel(self.state, self.config),
            # SaveModel(self.state, self.config),
            # LoadModel(self.state, self.config, view=True),
            # EvaluateModel(self.state, self.config),
            # VisualiseEvaluation(self.state, self.config),
        ]
        self.exe._execute_steps(steps, stage="parent")

    def __call__(self):
        self.run()
