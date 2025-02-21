from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    ai_models_dir : Path = Path("ai_models")
    debate_models_list_path: Path = field(init=False)
    judge_models_list_path: Path = field(init=False)
    topic_dir: Path = Path("topics")
    topic_list_path: Path = field(init=False)

    prompts_dir: Path = Path("prompts")
    prompts_path_yaml: Path = field(init=False)


    def __post_init__(self):
        self.ai_models_dir.mkdir(exist_ok=True)
        self.topic_dir.mkdir(exist_ok=True)
        self.prompts_dir.mkdir(exist_ok=True)

        self.debate_models_list_path = self.ai_models_dir / "debate_models.json"
        self.judge_models_list_path = self.ai_models_dir / "judge_models.json"
        self.topic_list_path = self.topic_dir / "topics_list.json"

        self.prompts_path_yaml = self.prompts_dir / "debate_prompts.yaml"
