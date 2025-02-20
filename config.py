from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    models_list_path: Path
    topic_list_path: Path
