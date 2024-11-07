from __future__ import annotations

from config.state_init import StateManager


class DatabaseFactory:
    """
    Factory for creating and configuring database components based on data type, mode, and stage.
    """

    def __init__(self, state: StateManager, stage: str):
        self.state = state
        self.stage = stage
        self.base_path = f"{state.api_config.symbol}_{state.api_config.mode}"

        self.db_ops = self.state.db_manager.ops
        self.data_handler = self.state.db_manager.handler

    def create_paths(self):
        """Create the load and save paths based on the stage."""
        if self.stage == "load1":
            load_path = f"{self.base_path}_transform"
            save_path = None
        elif self.stage == "load2":
            load_path = None
            save_path = f"{self.base_path}_fetch"
        else:
            raise ValueError(f"Invalid stage: {self.stage}. Expected 'load1' or 'load2'.")
        return load_path, save_path

    def create_steps(self):
        """Create the ETL steps based on the stage."""
        if self.stage == "load1":
            # Load1: Create table and insert data
            return [
                (self.db_ops.create_table_if_not_exists, self.create_paths()[0], None),
                (self.data_handler.insert_batches_to_db, self.create_paths()[0], None),
            ]
        elif self.stage == "load2":
            # Load2: Fetch data
            return [
                (self.data_handler.fetch_data, None, self.create_paths()[1]),
            ]
        else:
            raise ValueError(f"Invalid stage: {self.stage}. Expected 'load1' or 'load2'.")
