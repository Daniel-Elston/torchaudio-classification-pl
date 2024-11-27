from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

paths_store = {
    "raw": Path("data/raw/SpeechCommands/speech_commands_v0.02"),
    "hdf5": Path("data/audio_data.hdf5"),
    "processed": Path("data/processed"),
    "models": Path("models/cnn_model.pth"),
    "reports": Path("reports/figures"),
    "results": Path("reports/results/eval_results.json"),
    "mappings": Path("reports/mappings.json"),
}


@dataclass
class PathsConfig:
    paths: Dict[str, Path] = field(default_factory=dict)

    def __post_init__(self):
        self.paths = {k: Path(v) for k, v in paths_store.items()}

    def get_path(self, key: Optional[Union[str, Path]]) -> Optional[Path]:
        if key is None:
            return None
        if isinstance(key, Path):
            return key
        return self.paths.get(key)

    def validate_paths(self):
        for name, path in self.paths.items():
            if not path.exists():
                logging.warning(f"Path {name} does not exist: {path}")
