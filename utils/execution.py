from __future__ import annotations

from typing import Callable

from config.state_init import StateManager


class TaskExecutor:
    def __init__(self, state: StateManager):
        self.data_config = state.data_config
        self.paths = state.paths

    def run_main_step(self, step: Callable, *args, **kwargs):
        """Pipeline runner for main pipelines scripts (main.py)"""
        return step(*args, **kwargs)

    def run_parent_step(self, step: Callable, *args, **kwargs):
        """Pipeline runner for parent pipelines scripts (src/pipelines/*)"""
        return step(*args, **kwargs)

    @staticmethod
    def run_child_step(step: Callable, *args, **kwargs):
        """Pipeline runner for child pipelines scripts (lowest level scripts)"""
        return step(*args, **kwargs)

    def _execute_steps(self, steps, stage=None):
        if stage == "main":
            for step in steps:
                self.run_main_step(step)
        elif stage == "parent":
            for step in steps:
                self.run_parent_step(step)
