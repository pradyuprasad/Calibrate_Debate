from core.models import DebateTotal, SpeechType
from config import Config
from ai_models.load_models import load_judge_models
from pathlib import Path
from dotenv import load_dotenv
import logging

load_dotenv()


def checkIfComplete(path: Path) -> bool:
    try:
        # Load the debate from the json file
        debate = DebateTotal.load_from_json(path)

        # Check all speeches in both proposition and opposition outputs
        for speech_type in SpeechType:
            # Check proposition speeches
            if debate.proposition_output.speeches[speech_type] == -1:
                return False

            # Check opposition speeches
            if debate.opposition_output.speeches[speech_type] == -1:
                return False

        return True

    except Exception as e:
        # Handle any errors in loading or processing the file
        logging.error(f"Error checking debate completion status: {str(e)}")
        return False


def sanitize_model_name(model_name: str) -> str:
    """Convert model name to a valid filename by replacing / with _"""
    return model_name.replace("/", "_")


def run_judge_reliability_test(debate_paths: list[Path]):
    """
    Run each judge model 3 times on each debate.
    """
    config = Config()
    judge_models = load_judge_models(config)

    for debate_path in debate_paths:
        logging.info(f"Processing debate: {debate_path}")
        isComplete = checkIfComplete(debate_path)
        if not isComplete:
            print(f"{debate_path} is not complete")
            continue
        # Load the debate
        debate = DebateTotal.load_from_json(debate_path)

        # For each judge model, run 3 times
        for judge_model in judge_models:
            safe_model_name = sanitize_model_name(judge_model)
            for run in range(3):
                logging.info(f"Running {judge_model} - attempt {run + 1}")

                try:
                    # Get judgment for this run
                    config.judgement_processor.get_judgement_response(
                        debate=debate, model=judge_model
                    )

                    # Save after each judgment using sanitized model name
                    save_path = (
                        config.sample_judgments_dir
                        / f"{debate_path.stem}_judge_{safe_model_name}_run_{run + 1}.json"
                    )
                    debate.path_to_store = save_path
                    debate.save_to_json()

                except Exception as e:
                    logging.error(
                        f"Error getting judgment from {judge_model} (run {run + 1}): {e}"
                    )


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config = Config()

    debate_paths = list(config.sample_debates_dir.glob("*.json"))

    run_judge_reliability_test(debate_paths)
