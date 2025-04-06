import yaml

from config import Config
from core.models import DebatePrompts


def get_debate_prompt(config: Config) -> DebatePrompts:
    with open(config.prompts_path_yaml, "r") as file:
        prompts = yaml.safe_load(file)

    debator_prompts = DebatePrompts(
        first_speech_prompt=prompts["first_speech"],
        rebuttal_speech_prompt=prompts["rebuttal_speech"],
        final_speech_prompt=prompts["final_speech"],
        judge_prompt=prompts["judging_prompt"],
    )

    return debator_prompts
