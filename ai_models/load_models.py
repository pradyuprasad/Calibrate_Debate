from config import Config
from typing import List, Dict
import json

def load_debate_models(config: Config) -> Dict[str, List[float]]:
    with open(config.debate_models_list_path) as f:
        models_list = json.load(f)

    return models_list

def load_judge_models(config: Config) -> Dict[str, List[float]]:
    with open(config.judge_models_list_path) as f:
            models_list = json.load(f)

    return models_list
