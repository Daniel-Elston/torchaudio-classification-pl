from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pprint import pformat

from config.api import ApiConfig
from src.db.db_components import DatabaseConnection, DatabaseOperations, DataHandler


def db_creds():
    admin_conf = {
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "host": os.getenv("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT"),
    }
    db_conf = {
        "database": os.getenv("POSTGRES_DB"),
        "schema": os.getenv("POSTGRES_SCHEMA"),
        # "table": os.getenv("POSTGRES_TABLE"),
    }
    return admin_conf, db_conf


@dataclass
class DatabaseConfig(ApiConfig):
    stage: str = "load1"  # "load1" or "load2"
    admin_creds: dict = field(init=False)
    db_info: dict = field(init=False)

    database: str = field(init=False)
    schema: str = field(init=False)
    # table: str = field(init=False)
    table: str = field(init=False)

    overwrite: bool = False

    chunk_size: int = 5_00_000
    batch_size: int = 100_000

    def __post_init__(self):
        self.admin_creds, self.db_info = db_creds()
        self.database = self.db_info["database"]
        self.schema = self.db_info["schema"]
        # self.table = self.db_info["table"]
        self.table = self.create_table_config()

        logging.debug(f"Initialised DatabaseConfig:\n{pformat(self.__dict__)}\n")

    def create_table_config(self):
        """Dynamically create table name based on stage"""
        if self.stage == "load1":
            table_name = f"{self.symbol}_{self.mode}_transform"
            return table_name.lower()
        elif self.stage == "load2":
            table_name = f"{self.symbol}_{self.mode}_fetch"
            return table_name.lower()
        else:
            raise ValueError(f"Invalid stage: {self.stage}. Expected 'load1' or 'load2'.")


@dataclass
class DatabaseConnManager:
    config: DatabaseConfig
    conn: DatabaseConnection = field(init=False)
    ops: DatabaseOperations = field(init=False)
    handler: DataHandler = field(init=False)

    def __post_init__(self):
        self.initialize_database()

    def initialize_database(self):
        self.conn = DatabaseConnection(self.config.admin_creds, self.config.db_info)
        self.ops = DatabaseOperations(self.conn, self.config.schema, self.config.table)
        self.handler = DataHandler(
            self.conn, self.config.schema, self.config.table, self.config.batch_size
        )
