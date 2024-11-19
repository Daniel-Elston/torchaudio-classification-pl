from __future__ import annotations

from config.state_init import StateManager
from utils.execution import TaskExecutor

from src.data.load_imgs import HDF5DataModule
from src.models.init_model import InitialiseModel
from src.models.train import ModelTrainer
from src.models.validate import ValidateModel
from src.models.save_model import SaveModel
from src.models.load_model import LoadModel
from src.models.evaluate import EvaluateModel
from src.plots.vis_eval import VisualiseEvaluation
from src.data.label_mapping import CreateLabelMapping

class ModelPipeline:
    def __init__(self, state: StateManager, exe: TaskExecutor):
        self.state = state
        self.exe = exe
        self.config = state.model_config
        self.data_config = state.data_config

    def run(self):
        steps = [
            CreateLabelMapping(self.state, self.config),
            ModelTrainer(self.state),
            # ValidateModel(self.state, self.config),
            # SaveModel(self.state, self.config),
            # LoadModel(self.state, self.config, view=True),
            # EvaluateModel(self.state, self.config),
            # VisualiseEvaluation(self.state, self.config),
        ]
        self.exe._execute_steps(steps, stage="parent")

    def __call__(self):
        self.run()
