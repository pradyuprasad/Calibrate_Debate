from core.debate import get_judgement
from core.models import DebateTotal
from config import Config
from ai_models.load_models import load_judge_models
from pathlib import Path
from dotenv import load_dotenv
import logging

load_dotenv()

def sanitize_model_name(model_name: str) -> str:
    """Convert model name to a valid filename by replacing / with _"""
    return model_name.replace('/', '_')

def run_judge_reliability_test(debate_paths: list[Path]):
    """
    Run each judge model 3 times on each debate.
    """
    config = Config()
    judge_models = load_judge_models(config)

    for debate_path in debate_paths:
        logging.info(f"Processing debate: {debate_path}")

        # Load the debate
        debate = DebateTotal.load_from_json(debate_path)

        # For each judge model, run 3 times
        for judge_model in judge_models:
            safe_model_name = sanitize_model_name(judge_model)
            for run in range(3):
                logging.info(f"Running {judge_model} - attempt {run+1}")

                try:
                    # Get judgment for this run
                    get_judgement(
                        debate=debate,
                        prompts=debate.prompts,
                        judge_model=judge_model
                    )

                    # Save after each judgment using sanitized model name
                    save_path = debate_path.parent / f"{debate_path.stem}_judge_{safe_model_name}_run_{run+1}.json"
                    debate.path_to_store = save_path
                    debate.save_to_json()

                except Exception as e:
                    logging.error(f"Error getting judgment from {judge_model} (run {run+1}): {e}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Your 3 successful debates
    debate_paths = [
        Path("sample_debate_1.json"),
        Path("sample_debate_2.json"),
        Path("sample_debate_3.json")
    ]

    run_judge_reliability_test(debate_paths)
