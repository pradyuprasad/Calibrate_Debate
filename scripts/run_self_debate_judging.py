#!/usr/bin/env python3
"""
Script to analyze private self-debates by randomly selecting half of each model's debates
and judging them with the tournament jury configuration.
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from config import Config
from core.logger import LoggerFactory
from core.models import DebateTotal, JudgeResult


def setup_logging():
    """Configure logging for the script."""
    logger = LoggerFactory.get_logger(name="self_debate_judge", log_file="self_debate_judge.log")
    return logger


def find_model_debates(
    self_debates_dir: Path, logger: logging.Logger
) -> Dict[str, List[Path]]:
    """
    Find all self-debates and group them by model.

    Args:
        self_debates_dir: Directory containing the self-debates
        logger: Logger instance

    Returns:
        Dictionary mapping model names to lists of debate paths
    """
    model_debates = {}
    debate_paths = list(self_debates_dir.glob("*.json"))

    for path in debate_paths:
        try:
            debate = DebateTotal.load_from_json(path)

            # Verify this is a self-debate
            if debate.proposition_model == debate.opposition_model:
                model_name = debate.proposition_model
                if model_name not in model_debates:
                    model_debates[model_name] = []
                model_debates[model_name].append(path)
        except Exception as e:
            logger.error(f"Error loading debate {path}: {e}")

    # Log the results
    for model, debates in model_debates.items():
        logger.info(f"Found {len(debates)} self-debates for {model}")

    return model_debates


def select_random_debates(
    debates: List[Path], count: int, logger: logging.Logger
) -> List[Path]:
    """
    Randomly select a specified number of debates from the list.

    Args:
        debates: List of debate paths
        count: Number of debates to select
        logger: Logger instance

    Returns:
        List of randomly selected debate paths
    """
    if len(debates) <= count:
        logger.warning(
            f"Requested {count} debates but only {len(debates)} are available. Using all available debates."
        )
        return debates

    selected = random.sample(debates, count)
    logger.info(f"Randomly selected {len(selected)} debates for analysis")

    return selected


def judge_debate(
    debate_path: Path, config: Config, judge_models: List[str], voting_rounds: int, logger: logging.Logger
) -> Optional[List[JudgeResult]]:
    """
    Judge a debate using the tournament jury configuration.

    Args:
        debate_path: Path to the debate JSON file
        config: Config object with judgment processor
        judge_models: List of judge models to use
        voting_rounds: Number of voting rounds
        logger: Logger instance

    Returns:
        List of judgment results or None if judging failed
    """
    logger.info(f"Judging debate: {debate_path}")

    try:
        # Continue the debate to ensure it's complete
        debate = config.debate_service.continue_debate(debate_path)
        all_judgments: List[JudgeResult] = []

        # Process judgments from each judge for each voting round
        for i in range(voting_rounds):
            for judge_model in judge_models:
                judgment = config.judgement_processor.process_judgment(
                    debate=debate, model=judge_model
                )
                all_judgments.append(judgment)

        return all_judgments
    except Exception as e:
        logger.error(f"Judging failed: {str(e)}")
        return None


def find_winner(
    judgments: List[JudgeResult], logger: logging.Logger
) -> Tuple[str, float]:
    """
    Determine the winner based on judgment results.

    Args:
        judgments: List of judgment results
        logger: Logger instance

    Returns:
        Tuple of (winner, margin)
    """
    # Calculate confidence sums for each side
    prop_confidence = sum(
        judge.confidence
        for judge in judgments
        if judge.winner == "proposition"
    )
    opp_confidence = sum(
        judge.confidence
        for judge in judgments
        if judge.winner == "opposition"
    )
    total_confidence = prop_confidence + opp_confidence

    # Determine winner by total confidence
    if prop_confidence > opp_confidence:
        winner = "proposition"
        winner_confidence = prop_confidence
        loser_confidence = opp_confidence
    elif opp_confidence > prop_confidence:
        winner = "opposition"
        winner_confidence = opp_confidence
        loser_confidence = prop_confidence
    else:
        # In case of tie in confidence, random selection
        winner = random.choice(["proposition", "opposition"])
        winner_confidence = prop_confidence  # Equal to opp_confidence in this case
        loser_confidence = prop_confidence  # Equal to winner_confidence in this case
        logger.warning(
            "Debate resulted in a tie in confidence - randomly selecting winner"
        )

    # Calculate margin as winner confidence - loser confidence
    if total_confidence > 0:
        margin = (winner_confidence - loser_confidence) / total_confidence
    else:
        margin = 0.0

    return (winner, margin)


def analyze_self_debates(self_debates_dir: str):
    """
    Main function to analyze self-debates.

    Args:
        self_debates_dir: Directory containing the self-debates
        output_dir: Directory to save the results
    """
    # Setup
    logger = setup_logging()
    load_dotenv()
    config = Config()

    self_debates_path = Path(self_debates_dir)


    # Tournament judge configuration
    judge_models = [
        "qwen/qwq-32b",
        "google/gemini-pro-1.5",
        "deepseek/deepseek-chat",
    ]
    voting_rounds = 2

    logger.info(f"Using {len(judge_models)} judges for evaluation: {', '.join(judge_models)}")

    # Find all self-debates grouped by model
    model_debates = find_model_debates(self_debates_path, logger)

    # Process each model's debates
    results = {}

    for model, debates in model_debates.items():
        logger.info(f"Processing model: {model}")

        # Select random debates
        selected_debates = select_random_debates(debates, 3, logger)

        model_results = []

        # Judge each selected debate
        for debate_path in selected_debates:
            judgments = judge_debate(
                debate_path, config, judge_models, voting_rounds, logger
            )

            if judgments:
                winner, margin = find_winner(judgments, logger)

                # Load the debate to get the topic
                debate = DebateTotal.load_from_json(debate_path)
                topic = debate.motion.topic_description

                # Record the result
                debate_result = {
                    "debate_path": str(debate_path),
                    "topic": topic,
                    "winner": winner,
                    "margin": margin,
                    "judgments": [judgment.to_dict() for judgment in judgments],
                }

                model_results.append(debate_result)
                logger.info(
                    f"{model} self-debate on '{topic}': {winner} won with margin {margin:.2f}"
                )

        # Store results for this model
        results[model] = {
            "debates_analyzed": len(model_results),
            "results": model_results,
        }

    # Save overall results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze self-debates")
    parser.add_argument("--debates", default="private_self_debates", help="Directory containing self-debates")

    args = parser.parse_args()

    analyze_self_debates(args.debates)
