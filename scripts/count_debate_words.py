from typing import Dict, Tuple

import numpy as np

from config import Config
from core.models import DebateTotal


def analyze_debate_token_counts(config: Config) -> Dict[str, Tuple[float, float]]:
    json_files = list(config.sample_debates_dir.glob("*.json"))

    model_prompt_tokens = []
    model_completion_tokens = []

    for debate_file in json_files:
        debate = DebateTotal.load_from_json(debate_file)

        for model, usage in debate.debator_token_counts.model_usages.items():
            prompt_tokens = usage.successful_prompt_tokens + usage.failed_prompt_tokens
            completion_tokens = (
                usage.successful_completion_tokens + usage.failed_completion_tokens
            )

            model_prompt_tokens.append(prompt_tokens)
            model_completion_tokens.append(completion_tokens)

    prompt_mean = float(np.mean(model_prompt_tokens))
    prompt_var = float(np.var(model_prompt_tokens))
    completion_mean = float(np.mean(model_completion_tokens))
    completion_var = float(np.var(model_completion_tokens))

    stats = {
        "prompt_tokens": (prompt_mean, prompt_var),
        "completion_tokens": (completion_mean, completion_var),
    }

    return stats


if __name__ == "__main__":
    config = Config()
    stats = analyze_debate_token_counts(config)
    print("\nPer-model statistics:")
    print(
        f"Prompt tokens - mean: {stats['prompt_tokens'][0]:,.0f}, variance: {stats['prompt_tokens'][1]:,.0f}"
    )
    print(
        f"Completion tokens - mean: {stats['completion_tokens'][0]:,.0f}, variance: {stats['completion_tokens'][1]:,.0f}"
    )
