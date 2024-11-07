from __future__ import annotations

import pandas as pd

from config.state_init import StateManager
from utils.execution import TaskExecutor


class NormaliseCryptoLive:
    """Load dataset and perform base processing"""

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        steps = [
            self.access_nested,
        ]
        for step in steps:
            df = TaskExecutor.run_child_step(step, df)
        return df

    def access_nested(self, df: pd.DataFrame) -> pd.DataFrame:
        df = pd.json_normalize(df.to_dict(orient="records"), sep="_")
        return df


class NormaliseCryptoHistorical:
    """Load dataset and perform base processing"""

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        steps = [
            self.access_ohlcv,
            self.assign_headings,
        ]
        for step in steps:
            df = TaskExecutor.run_child_step(step, df)
        return df

    def access_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def assign_headings(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = ["datetime", "open", "high", "low", "close", "volume"]
        return df


class NormaliseStock:
    """Load dataset and perform base processing"""

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        steps = [self.process_data, self.normalise_data, self.sort_data]
        for step in steps:
            df = TaskExecutor.run_child_step(step, df)
        return df

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.iloc[6:, :]
        df = df.drop("Meta Data", axis=1)
        return df

    def normalise_data(self, df: pd.DataFrame) -> pd.DataFrame:
        datetime = df.index
        df = pd.json_normalize(df.to_dict(orient="records"), sep="_")
        df["datetime"] = datetime
        return df

    def sort_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = ["open", "high", "low", "close", "volume", "datetime"]
        df = df[["datetime", "open", "high", "low", "close", "volume"]]
        return df
