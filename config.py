from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    # Existing paths
    ai_models_dir: Path = Path("ai_models")
    debate_models_list_path: Path = field(init=False)
    judge_models_list_path: Path = field(init=False)
    topic_dir: Path = Path("topics")
    topic_list_path: Path = field(init=False)
    prompts_dir: Path = Path("prompts")
    prompts_path_yaml: Path = field(init=False)

    # New output paths
    outputs_dir: Path = Path("outputs")
    debates_dir: Path = field(init=False)
    judgments_dir: Path = field(init=False)

    # Sample paths
    samples_dir: Path = Path("samples")
    sample_debates_dir: Path = field(init=False)
    sample_judgments_dir: Path = field(init=False)

    # Tournament paths
    tournament_dir: Path = Path("tournament")
    tournament_results_path: Path = field(init=False)
    num_rounds: int = 3  # Number of Swiss rounds
    k_factor: int = 64  # For elo calculations


    def __post_init__(self):
        # Existing directory creation
        self.ai_models_dir.mkdir(exist_ok=True)
        self.topic_dir.mkdir(exist_ok=True)
        self.prompts_dir.mkdir(exist_ok=True)

        # New output directories
        self.outputs_dir.mkdir(exist_ok=True)
        self.debates_dir = self.outputs_dir / "debates"
        self.judgments_dir = self.outputs_dir / "judgments"
        self.debates_dir.mkdir(exist_ok=True)
        self.judgments_dir.mkdir(exist_ok=True)

        # Sample directories
        self.samples_dir.mkdir(exist_ok=True)
        self.sample_debates_dir = self.samples_dir / "debates"
        self.sample_judgments_dir = self.samples_dir / "judgments"
        self.sample_debates_dir.mkdir(exist_ok=True)
        self.sample_judgments_dir.mkdir(exist_ok=True)

        # Tournament directories
        self.tournament_dir.mkdir(exist_ok=True)
        self.tournament_results_path = self.tournament_dir / "tournament_results.json"

        # Existing path assignments
        self.debate_models_list_path = self.ai_models_dir / "debate_models.json"
        self.judge_models_list_path = self.ai_models_dir / "judge_models.json"
        self.topic_list_path = self.topic_dir / "topics_list.json"
        self.prompts_path_yaml = self.prompts_dir / "debate_prompts.yaml"
