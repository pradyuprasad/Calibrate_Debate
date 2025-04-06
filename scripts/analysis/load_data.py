import logging
from pathlib import Path
from typing import List

from core.models import DebateTotal, Side
# Import Pydantic models
from scripts.analysis.models import (DebateData, DebateResults, ModelStats,
                                     TopicStats)


def load_debate_data() -> DebateResults:
    """
    Loads and processes all debate data from a hardcoded private tournament path.

    Returns:
        DebateResults object containing debate data and statistics
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("debate_analysis")

    # Hardcoded path to private debate tournament
    PRIVATE_TOURNAMENT_PATH = "private_bet_tournament"

    # Load all debates
    logger.info(f"Loading debates from {PRIVATE_TOURNAMENT_PATH}")
    debates = extract_debates_from_private_tournament(Path(PRIVATE_TOURNAMENT_PATH))
    logger.info(f"Loaded {len(debates)} debates total")

    # Process the data
    result = process_debate_data(debates)

    return result

def extract_debates_from_private_tournament(private_tournament_dir: Path) -> List[DebateData]:
    """
    Loads all debates from the private tournament directory and converts them to standardized format.

    Args:
        private_tournament_dir: Path object pointing to the private tournament directory

    Returns:
        List of standardized DebateData objects
    """
    # Initialize logger
    logger = logging.getLogger("debate_extraction")

    # Initialize debates list
    debates = []

    # Find all debate JSON files
    for debate_path in private_tournament_dir.glob("*.json"):
        # Skip any results files
        if "results.json" in debate_path.name:
            continue

        try:
            debate = DebateTotal.load_from_json(debate_path)

            # Process judge results
            winner_counts = {"proposition": 0, "opposition": 0}
            for result in debate.judge_results:
                winner_counts[result.winner] += 1

            winner = "proposition" if winner_counts["proposition"] > winner_counts["opposition"] else "opposition"
            total_judges = len(debate.judge_results)
            max_agreement = max(winner_counts.values())
            judge_agreement = "unanimous" if max_agreement == total_judges else "split"

            # Process bets by speech type
            prop_bets = {}
            opp_bets = {}
            for bet in debate.debator_bets:
                if bet.side == Side.PROPOSITION:
                    prop_bets[bet.speech_type.value] = bet.amount
                elif bet.side == Side.OPPOSITION:
                    opp_bets[bet.speech_type.value] = bet.amount

            # Create standardized debate object
            debate_data = DebateData(
                id=debate_path.stem,
                round=0,  # No longer using rounds
                tournament="private_tournament",  # Single tournament
                path=str(debate_path),
                proposition_model=debate.proposition_model,
                opposition_model=debate.opposition_model,
                topic=debate.motion.topic_description,
                winner=winner,
                judge_results=winner_counts,
                prop_bets=prop_bets,
                opp_bets=opp_bets,
                judge_agreement=judge_agreement
            )

            debates.append(debate_data)
            logger.info(f"Loaded debate: {debate_path.name}")

        except Exception as e:
            logger.error(f"Failed to load debate {debate_path}: {str(e)}")

    logger.info(f"Extracted {len(debates)} debates total")
    return debates


def process_debate_data(debates: List[DebateData]) -> DebateResults:
    """
    Processes a list of debates to extract statistics.

    Args:
        debates: List of DebateData objects

    Returns:
        DebateResults object containing debate data and statistics
    """
    # Initialize data structures
    model_stats = {}
    topic_stats = {}

    # Extract information from debates
    for debate in debates:
        # Update model statistics
        for model_name in [debate.proposition_model, debate.opposition_model]:
            if model_name not in model_stats:
                model_stats[model_name] = ModelStats()

            model_stats[model_name].debates += 1

            # Update wins/losses
            if (debate.winner == "proposition" and model_name == debate.proposition_model) or \
               (debate.winner == "opposition" and model_name == debate.opposition_model):
                model_stats[model_name].wins += 1

                if model_name == debate.proposition_model:
                    model_stats[model_name].prop_wins += 1
                else:
                    model_stats[model_name].opp_wins += 1
            else:
                model_stats[model_name].losses += 1

                if model_name == debate.proposition_model:
                    model_stats[model_name].prop_losses += 1
                else:
                    model_stats[model_name].opp_losses += 1

        # Update topic statistics
        topic = debate.topic
        if topic not in topic_stats:
            topic_stats[topic] = TopicStats(models=[])

        topic_stats[topic].count += 1

        # Add models if not already in the list
        if debate.proposition_model not in topic_stats[topic].models:
            topic_stats[topic].models.append(debate.proposition_model)
        if debate.opposition_model not in topic_stats[topic].models:
            topic_stats[topic].models.append(debate.opposition_model)

        if debate.winner == "proposition":
            topic_stats[topic].prop_wins += 1
        else:
            topic_stats[topic].opp_wins += 1

        if debate.judge_agreement == "unanimous":
            topic_stats[topic].unanimous_decisions += 1
        else:
            topic_stats[topic].split_decisions += 1

    # Calculate win rates and other derived statistics for models
    for model_name, stats in model_stats.items():
        total_debates = stats.debates
        stats.win_rate = stats.wins / total_debates if total_debates > 0 else 0

        prop_debates = stats.prop_wins + stats.prop_losses
        stats.prop_win_rate = stats.prop_wins / prop_debates if prop_debates > 0 else 0

        opp_debates = stats.opp_wins + stats.opp_losses
        stats.opp_win_rate = stats.opp_wins / opp_debates if opp_debates > 0 else 0

    # Calculate derived statistics for topics
    for topic, stats in topic_stats.items():
        total_debates = stats.count
        stats.prop_win_rate = stats.prop_wins / total_debates if total_debates > 0 else 0
        stats.unanimous_rate = stats.unanimous_decisions / total_debates if total_debates > 0 else 0

    # Return the comprehensive data structure
    return DebateResults(
        debates=debates,
        model_stats=model_stats,
        topic_stats=topic_stats
    )
