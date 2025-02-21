from core.debate import run_debate
from config import Config
from topics.load_topics import load_topics
from ai_models.load_models import load_debate_models
from prompts.load_prompts import get_debate_prompt
from pathlib import Path
from dotenv import load_dotenv
import random

load_dotenv()

def run_sample_debates(num_samples=4):
    config = Config()
    topics = load_topics(config)
    debate_models = load_debate_models(config)
    prompts = get_debate_prompt(config)

    # Randomly select topics and model pairs
    selected_topics = random.sample(topics, k=2)
    model_names = list(debate_models.keys())

    sample_debates = []
    for i, topic in enumerate(selected_topics):
        # Create two debates per topic with different model pairs
        for j in range(2):
            # Randomly select two different models
            debaters = random.sample(model_names, k=2)
            sample_debates.append({
                "topic": topic,
                "prop": debaters[0],
                "opp": debaters[1],
                "output": Path(f"sample_debate_{i*2 + j + 1}.json")
            })

    for debate in sample_debates:
        run_debate(
            proposition_model=debate["prop"],
            opposition_model=debate["opp"],
            motion=debate["topic"],
            prompts=prompts,
            path_to_store_debate=debate["output"],
            judge_models=[]
        )

if __name__ == "__main__":
    run_sample_debates()
