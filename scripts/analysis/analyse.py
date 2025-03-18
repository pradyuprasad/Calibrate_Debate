import json
import logging
from pathlib import Path
from collections import namedtuple
from typing import List, Dict, Optional, Any
from collections import defaultdict
import numpy as np
import statsmodels.api as sm
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, chi2_contingency, fisher_exact, wilcoxon, mannwhitneyu

from core.models import DebateTotal, Side

# Define a standardized debate structure for analysis
DebateData = namedtuple('DebateData', [
    'id', 'round', 'tournament', 'path',
    'proposition_model', 'opposition_model',
    'topic', 'winner', 'judge_results',
    'prop_bets', 'opp_bets', 'judge_agreement'
])

def load_tournament_data(tournament_dirs: List[Path], excluded_models: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load and process tournament data from multiple tournament directories.

    Args:
        tournament_dirs: List of Path objects pointing to tournament directories
        excluded_models: List of model names to exclude from analysis

    Returns:
        A dictionary containing:
            - debates: List of standardized debate objects
            - models: Set of model names included in analysis
            - topics: Set of unique debate topics
            - tournaments: Dict with metadata about each tournament
            - model_stats: Combined model statistics from all tournaments
    """
    # Initialize logger
    logger = logging.getLogger("tournament_analysis")

    # Initialize data structures
    debates = []
    models = set()
    topics = set()
    tournaments = {}
    model_stats = {}

    # Set default for excluded_models if None
    if excluded_models is None:
        excluded_models = []

    # Load tournament results from all directories
    for tournament_dir in tournament_dirs:
        tournament_id = tournament_dir.name
        tournaments[tournament_id] = {
            "path": str(tournament_dir),
            "debate_count": 0,
            "rounds": set()
        }

        # Try to load tournament statistics
        tournament_results_path = tournament_dir / "tournament_results.json"
        try:
            with open(tournament_results_path, "r") as f:
                tournament_results = json.load(f)
                # Merge model statistics, excluding any models in the excluded list
                for model, statistics in tournament_results.get("model_stats", {}).items():
                    if model not in excluded_models:
                        model_stats[model] = statistics
                logger.info(f"Loaded tournament results from: {tournament_dir}")
        except FileNotFoundError:
            logger.warning(f"Tournament results file not found: {tournament_results_path}")
        except json.JSONDecodeError:
            logger.error(f"Error parsing JSON from: {tournament_results_path}")

        # Find all debate files from this tournament directory
        round_dirs = [d for d in tournament_dir.glob("round_*") if d.is_dir()]

        for round_dir in round_dirs:
            round_num = int(round_dir.name.split("_")[1])
            tournaments[tournament_id]["rounds"].add(round_num)

            for debate_path in round_dir.glob("*.json"):
                try:
                    debate = DebateTotal.load_from_json(debate_path)

                    # Skip debates involving any excluded model
                    if (debate.proposition_model in excluded_models or
                        debate.opposition_model in excluded_models):
                        logger.info(f"Skipping debate with excluded model: {debate_path.name}")
                        continue

                    # Add models to the set of known models
                    models.add(debate.proposition_model)
                    models.add(debate.opposition_model)

                    # Add topic to the set of known topics
                    topics.add(debate.motion.topic_description)

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
                        round=round_num,
                        tournament=tournament_id,
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
                    tournaments[tournament_id]["debate_count"] += 1
                    logger.info(f"Loaded debate: {debate_path.name} from {tournament_id}")

                except Exception as e:
                    logger.error(f"Failed to load debate {debate_path}: {str(e)}")

    # Log summary statistics
    logger.info(f"Loaded {len(debates)} debates across {len(tournaments)} tournaments")
    logger.info(f"Found {len(models)} unique models and {len(topics)} unique topics")

    # Return a comprehensive data structure
    return {
        "debates": debates,
        "models": models,
        "topics": topics,
        "tournaments": tournaments,
        "model_stats": model_stats
    }


def analyze_win_rates_by_confidence(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Analyze win rates for different confidence tiers.

    Args:
        data: The tournament data dictionary from load_tournament_data

    Returns:
        Dictionary mapping confidence tiers to win statistics
    """
    win_rates = {
        "0-25": {"wins": 0, "total": 0},
        "26-50": {"wins": 0, "total": 0},
        "51-75": {"wins": 0, "total": 0},
        "76-100": {"wins": 0, "total": 0}
    }

    # Process each debate
    for debate in data["debates"]:
        # Check proposition confidence
        if "opening" in debate.prop_bets:
            prop_conf = debate.prop_bets["opening"]
            tier = get_confidence_tier(prop_conf)
            win_rates[tier]["total"] += 1
            if debate.winner == "proposition":
                win_rates[tier]["wins"] += 1

        # Check opposition confidence
        if "opening" in debate.opp_bets:
            opp_conf = debate.opp_bets["opening"]
            tier = get_confidence_tier(opp_conf)
            win_rates[tier]["total"] += 1
            if debate.winner == "opposition":
                win_rates[tier]["wins"] += 1

    # Calculate percentages
    results = {}
    for tier, counts in win_rates.items():
        if counts["total"] > 0:
            win_percentage = (counts["wins"] / counts["total"]) * 100
            results[tier] = {
                "wins": counts["wins"],
                "total": counts["total"],
                "percentage": win_percentage
            }
        else:
            results[tier] = {"wins": 0, "total": 0, "percentage": 0}

    return results

def get_confidence_tier(confidence: float) -> str:
    """Helper function to determine confidence tier."""
    if confidence <= 25:
        return "0-25"
    elif confidence <= 50:
        return "26-50"
    elif confidence <= 75:
        return "51-75"
    else:
        return "76-100"

def analyze_model_confidence_changes(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Analyze how model confidence changes throughout debates.

    Args:
        data: The tournament data dictionary from load_tournament_data

    Returns:
        Dictionary mapping models to confidence change statistics
    """
    # Initialize tracking structures
    model_changes = defaultdict(list)

    for debate in data["debates"]:
        # Process proposition confidence changes
        if "opening" in debate.prop_bets and "closing" in debate.prop_bets:
            prop_change = debate.prop_bets["closing"] - debate.prop_bets["opening"]
            model_changes[debate.proposition_model].append(prop_change)

        # Process opposition confidence changes
        if "opening" in debate.opp_bets and "closing" in debate.opp_bets:
            opp_change = debate.opp_bets["closing"] - debate.opp_bets["opening"]
            model_changes[debate.opposition_model].append(opp_change)

    # Calculate statistics for each model
    results = {}
    for model, changes in model_changes.items():
        if changes:
            avg_change = sum(changes) / len(changes)
            avg_abs_change = sum(abs(c) for c in changes) / len(changes)
            max_increase = max(changes)
            max_decrease = min(changes)

            results[model] = {
                "avg_change": avg_change,
                "avg_abs_change": avg_abs_change,
                "max_increase": max_increase,
                "max_decrease": max_decrease,
                "sample_size": len(changes)
            }

    return results

def analyze_judge_agreement(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze patterns in judge agreement across debates.

    Args:
        data: The tournament data dictionary from load_tournament_data

    Returns:
        Dictionary with judge agreement statistics
    """
    # Initialize counters
    agreement_counts = {"unanimous": 0, "split": 0}
    dissent_distribution = defaultdict(int)
    tournament_agreement = defaultdict(lambda: {"unanimous": 0, "split": 0})

    # Process each debate
    for debate in data["debates"]:
        # Update unanimous/split counts
        agreement_counts[debate.judge_agreement] += 1
        tournament_agreement[debate.tournament][debate.judge_agreement] += 1

        # Calculate number of dissenting judges
        if debate.judge_results:
            total_judges = debate.judge_results["proposition"] + debate.judge_results["opposition"]
            max_votes = max(debate.judge_results["proposition"], debate.judge_results["opposition"])
            dissenting_votes = total_judges - max_votes
            dissent_distribution[dissenting_votes] += 1

    # Calculate percentages
    total_debates = agreement_counts["unanimous"] + agreement_counts["split"]

    results = {
        "overall": {
            "unanimous": agreement_counts["unanimous"],
            "split": agreement_counts["split"],
            "total": total_debates,
            "unanimous_percent": (agreement_counts["unanimous"] / total_debates * 100) if total_debates > 0 else 0,
            "split_percent": (agreement_counts["split"] / total_debates * 100) if total_debates > 0 else 0
        },
        "dissent_distribution": dict(dissent_distribution),
        "by_tournament": {}
    }

    # Calculate per-tournament statistics
    for tournament, counts in tournament_agreement.items():
        tourney_total = counts["unanimous"] + counts["split"]
        if tourney_total > 0:
            results["by_tournament"][tournament] = {
                "unanimous": counts["unanimous"],
                "split": counts["split"],
                "total": tourney_total,
                "unanimous_percent": (counts["unanimous"] / tourney_total * 100),
                "split_percent": (counts["split"] / tourney_total * 100)
            }

    return results

def analyze_topic_difficulty(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Analyze topic difficulty based on judge disagreement and confidence changes.

    Args:
        data: The tournament data dictionary from load_tournament_data

    Returns:
        Dictionary mapping topics to difficulty metrics
    """
    # Initialize tracking structures
    topic_data = defaultdict(lambda: {
        "judge_disagreements": 0,
        "total_debates": 0,
        "confidence_changes": []
    })

    # Process each debate
    for debate in data["debates"]:
        topic = debate.topic
        topic_data[topic]["total_debates"] += 1

        # Count judge disagreements
        if debate.judge_agreement == "split":
            topic_data[topic]["judge_disagreements"] += 1

        # Track confidence changes
        if "opening" in debate.prop_bets and "closing" in debate.prop_bets:
            prop_change = abs(debate.prop_bets["closing"] - debate.prop_bets["opening"])
            topic_data[topic]["confidence_changes"].append(prop_change)

        if "opening" in debate.opp_bets and "closing" in debate.opp_bets:
            opp_change = abs(debate.opp_bets["closing"] - debate.opp_bets["opening"])
            topic_data[topic]["confidence_changes"].append(opp_change)

    # Calculate difficulty metrics for each topic
    results = {}
    for topic, statistics in topic_data.items():
        if statistics["total_debates"] > 0:
            # Calculate average confidence change
            avg_conf_change = (
                sum(statistics["confidence_changes"]) / len(statistics["confidence_changes"])
                if statistics["confidence_changes"] else 0
            )

            # Calculate judge disagreement percentage
            judge_disagreement_pct = (
                statistics["judge_disagreements"] / statistics["total_debates"] * 100
            )

            # Calculate overall difficulty index
            difficulty_index = avg_conf_change + judge_disagreement_pct

            results[topic] = {
                "total_debates": statistics["total_debates"],
                "judge_disagreements": statistics["judge_disagreements"],
                "judge_disagreement_percent": judge_disagreement_pct,
                "avg_confidence_change": avg_conf_change,
                "difficulty_index": difficulty_index
            }

    return results

def analyze_model_calibration(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze how well calibrated each model's confidence is with actual outcomes.

    Args:
        data: The tournament data dictionary from load_tournament_data

    Returns:
        Dictionary mapping models to calibration metrics
    """
    # Initialize tracking structures
    model_calibration = defaultdict(lambda: {"confidence": [], "win": []})

    # Process each debate
    for debate in data["debates"]:
        # Process proposition bets
        if "opening" in debate.prop_bets:
            model = debate.proposition_model
            confidence = debate.prop_bets["opening"]
            win = 1 if debate.winner == "proposition" else 0

            model_calibration[model]["confidence"].append(confidence)
            model_calibration[model]["win"].append(win)

        # Process opposition bets
        if "opening" in debate.opp_bets:
            model = debate.opposition_model
            confidence = debate.opp_bets["opening"]
            win = 1 if debate.winner == "opposition" else 0

            model_calibration[model]["confidence"].append(confidence)
            model_calibration[model]["win"].append(win)

    # Calculate calibration metrics for each model
    results = {}
    for model, data_points in model_calibration.items():
        if data_points["confidence"] and data_points["win"]:
            n = len(data_points["confidence"])

            # Calculate basic calibration score (mean squared error)
            calibration_score = sum([(data_points["confidence"][i]/100 - data_points["win"][i])**2
                                    for i in range(n)]) / n

            # Calculate average confidence and win rate
            avg_confidence = sum(data_points["confidence"]) / n
            win_rate = sum(data_points["win"]) / n * 100

            # Calculate overconfidence measure
            overconfidence = avg_confidence - win_rate

            # Calculate additional breakdowns
            # By confidence tier
            tiers = {"0-25": [], "26-50": [], "51-75": [], "76-100": []}
            for conf, win in zip(data_points["confidence"], data_points["win"]):
                tier = get_confidence_tier(conf)
                tiers[tier].append(win)

            tier_accuracy = {}
            for tier, wins in tiers.items():
                if wins:
                    tier_accuracy[tier] = {
                        "count": len(wins),
                        "win_rate": sum(wins) / len(wins) * 100
                    }

            results[model] = {
                "calibration_score": calibration_score,
                "avg_confidence": avg_confidence,
                "win_rate": win_rate,
                "overconfidence": overconfidence,
                "sample_size": n,
                "tier_accuracy": tier_accuracy
            }

    return results

def analyze_confidence_gaps(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the relationship between confidence gaps and debate outcomes.

    Args:
        data: The tournament data dictionary from load_tournament_data

    Returns:
        Dictionary with confidence gap analysis results
    """
    # Store gaps and outcomes
    confidence_gaps = []

    # Process each debate
    for debate in data["debates"]:
        if "opening" in debate.prop_bets and "opening" in debate.opp_bets:
            prop_conf = debate.prop_bets["opening"]
            opp_conf = debate.opp_bets["opening"]

            # Calculate absolute gap
            gap = abs(prop_conf - opp_conf)

            # Determine if higher confidence side won
            higher_conf_side = "proposition" if prop_conf > opp_conf else "opposition"
            higher_conf_won = higher_conf_side == debate.winner

            confidence_gaps.append({
                "gap": gap,
                "winner": debate.winner,
                "higher_conf_side": higher_conf_side,
                "higher_conf_won": higher_conf_won,
                "prop_conf": prop_conf,
                "opp_conf": opp_conf
            })

    # Skip further analysis if no gaps found
    if not confidence_gaps:
        return {"error": "No confidence gaps available for analysis"}

    # Calculate overall statistics
    avg_gap = sum(item["gap"] for item in confidence_gaps) / len(confidence_gaps)
    higher_conf_wins = sum(1 for item in confidence_gaps if item["higher_conf_won"])
    higher_conf_win_rate = (higher_conf_wins / len(confidence_gaps)) * 100

    # Group by gap size
    small_gaps = [item for item in confidence_gaps if item["gap"] <= 25]
    medium_gaps = [item for item in confidence_gaps if 25 < item["gap"] <= 50]
    large_gaps = [item for item in confidence_gaps if item["gap"] > 50]

    # Calculate statistics for each group
    gap_groups = {}
    for label, gaps in [("small", small_gaps), ("medium", medium_gaps), ("large", large_gaps)]:
        if gaps:
            wins = sum(1 for item in gaps if item["higher_conf_won"])
            gap_groups[label] = {
                "count": len(gaps),
                "avg_gap": sum(item["gap"] for item in gaps) / len(gaps),
                "higher_conf_wins": wins,
                "higher_conf_win_rate": (wins / len(gaps)) * 100
            }

    return {
        "overall": {
            "count": len(confidence_gaps),
            "avg_gap": avg_gap,
            "higher_conf_wins": higher_conf_wins,
            "higher_conf_win_rate": higher_conf_win_rate
        },
        "by_gap_size": gap_groups,
        "raw_data": confidence_gaps
    }

def analyze_overconfidence_metrics(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze overconfidence patterns for each model.

    Args:
        data: The tournament data dictionary from load_tournament_data

    Returns:
        Dictionary mapping models to overconfidence metrics
    """
    # Initialize tracking structures
    overconfidence_metrics = defaultdict(lambda: {
        "high_conf_debates": 0,
        "high_conf_losses": 0,
        "low_conf_debates": 0,
        "low_conf_wins": 0,
        "avg_conf_wins": [],
        "avg_conf_losses": []
    })

    # Process each debate
    for debate in data["debates"]:
        # Process proposition
        if "opening" in debate.prop_bets:
            model = debate.proposition_model
            conf = debate.prop_bets["opening"]
            won = debate.winner == "proposition"

            # Track high confidence debates
            if conf > 75:
                overconfidence_metrics[model]["high_conf_debates"] += 1
                if not won:
                    overconfidence_metrics[model]["high_conf_losses"] += 1

            # Track low confidence debates
            if conf < 25:
                overconfidence_metrics[model]["low_conf_debates"] += 1
                if won:
                    overconfidence_metrics[model]["low_conf_wins"] += 1

            # Track average confidence in wins/losses
            if won:
                overconfidence_metrics[model]["avg_conf_wins"].append(conf)
            else:
                overconfidence_metrics[model]["avg_conf_losses"].append(conf)

        # Process opposition
        if "opening" in debate.opp_bets:
            model = debate.opposition_model
            conf = debate.opp_bets["opening"]
            won = debate.winner == "opposition"

            # Track high confidence debates
            if conf > 75:
                overconfidence_metrics[model]["high_conf_debates"] += 1
                if not won:
                    overconfidence_metrics[model]["high_conf_losses"] += 1

            # Track low confidence debates
            if conf < 25:
                overconfidence_metrics[model]["low_conf_debates"] += 1
                if won:
                    overconfidence_metrics[model]["low_conf_wins"] += 1

            # Track average confidence in wins/losses
            if won:
                overconfidence_metrics[model]["avg_conf_wins"].append(conf)
            else:
                overconfidence_metrics[model]["avg_conf_losses"].append(conf)

    # Calculate metrics for each model
    results = {}
    for model, metrics in overconfidence_metrics.items():
        avg_win_conf = sum(metrics["avg_conf_wins"]) / len(metrics["avg_conf_wins"]) if metrics["avg_conf_wins"] else 0
        avg_loss_conf = sum(metrics["avg_conf_losses"]) / len(metrics["avg_conf_losses"]) if metrics["avg_conf_losses"] else 0

        # Calculate high confidence loss rate
        high_conf_loss_rate = 0
        if metrics["high_conf_debates"] > 0:
            high_conf_loss_rate = (metrics["high_conf_losses"] / metrics["high_conf_debates"]) * 100

        # Calculate low confidence win rate
        low_conf_win_rate = 0
        if metrics["low_conf_debates"] > 0:
            low_conf_win_rate = (metrics["low_conf_wins"] / metrics["low_conf_debates"]) * 100

        # Calculate confidence accuracy ratio
        conf_accuracy_ratio = 0
        if avg_loss_conf > 0:
            conf_accuracy_ratio = avg_win_conf / avg_loss_conf

        results[model] = {
            "high_conf_debates": metrics["high_conf_debates"],
            "high_conf_losses": metrics["high_conf_losses"],
            "high_conf_loss_rate": high_conf_loss_rate,
            "low_conf_debates": metrics["low_conf_debates"],
            "low_conf_wins": metrics["low_conf_wins"],
            "low_conf_win_rate": low_conf_win_rate,
            "avg_win_confidence": avg_win_conf,
            "avg_loss_confidence": avg_loss_conf,
            "confidence_accuracy_ratio": conf_accuracy_ratio
        }

    return results

def run_statistical_tests(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run statistical hypothesis tests on the tournament data.

    Args:
        data: The tournament data dictionary from load_tournament_data

    Returns:
        Dictionary with statistical test results
    """
    # Prepare data structures for statistical tests
    all_opening_confidences = []
    all_win_outcomes = []
    prop_opening_confidences = []
    prop_win_outcomes = []
    opp_opening_confidences = []
    opp_win_outcomes = []

    # Contingency table for side vs. outcome
    side_outcome_counts = {
        "proposition": {"win": 0, "loss": 0},
        "opposition": {"win": 0, "loss": 0}
    }

    # Process each debate
    for debate in data["debates"]:
        # Get opening confidences
        prop_conf = debate.prop_bets.get("opening")
        opp_conf = debate.opp_bets.get("opening")

        # Get win outcomes
        prop_win = 1 if debate.winner == "proposition" else 0
        opp_win = 1 if debate.winner == "opposition" else 0

        # Update data structures if we have both confidences
        if prop_conf is not None and opp_conf is not None:
            # Update proposition data
            prop_opening_confidences.append(prop_conf)
            prop_win_outcomes.append(prop_win)

            # Update opposition data
            opp_opening_confidences.append(opp_conf)
            opp_win_outcomes.append(opp_win)

            # Update all data
            all_opening_confidences.extend([prop_conf, opp_conf])
            all_win_outcomes.extend([prop_win, opp_win])

            # Update contingency table
            if prop_win:
                side_outcome_counts["proposition"]["win"] += 1
                side_outcome_counts["opposition"]["loss"] += 1
            else:
                side_outcome_counts["proposition"]["loss"] += 1
                side_outcome_counts["opposition"]["win"] += 1

    results = {}

    # Skip tests if insufficient data
    if len(all_opening_confidences) < 5:
        return {"error": "Insufficient data for statistical testing"}

    # Test 1: General Overconfidence
    # One-sample t-test for confidence > 50
    t_stat, p_value = ttest_1samp(all_opening_confidences, 50)
    one_tailed_p = p_value / 2 if t_stat > 0 else 1 - (p_value / 2)

    results["overconfidence_t_test"] = {
        "mean_confidence": np.mean(all_opening_confidences),
        "t_statistic": t_stat,
        "p_value": p_value,
        "one_tailed_p": one_tailed_p,
        "significant": one_tailed_p < 0.05,
        "interpretation": "Statistically significant overconfidence" if one_tailed_p < 0.05 else "No significant overconfidence"
    }

    # Test 2: Paired t-test for confidence vs. outcomes
    paired_t_stat, paired_p_value = ttest_rel(
        all_opening_confidences,
        [x*100 for x in all_win_outcomes]
    )
    paired_one_tailed_p = paired_p_value / 2 if paired_t_stat > 0 else 1 - (paired_p_value / 2)

    mean_diff = np.mean(np.array(all_opening_confidences) - np.array(all_win_outcomes)*100)

    results["confidence_vs_outcome_t_test"] = {
        "mean_confidence": np.mean(all_opening_confidences),
        "mean_win_rate": np.mean(all_win_outcomes)*100,
        "mean_difference": mean_diff,
        "t_statistic": paired_t_stat,
        "p_value": paired_p_value,
        "one_tailed_p": paired_one_tailed_p,
        "significant": paired_one_tailed_p < 0.05,
        "interpretation": "Statistically significant overconfidence" if paired_one_tailed_p < 0.05 else "No significant overconfidence"
    }

    # Test 3: Side advantage testing
    # Create contingency table for chi-square test
    contingency_table = [
        [side_outcome_counts["proposition"]["win"], side_outcome_counts["proposition"]["loss"]],
        [side_outcome_counts["opposition"]["win"], side_outcome_counts["opposition"]["loss"]]
    ]

    chi2, chi2_p_value, dof, expected = chi2_contingency(contingency_table)

    prop_win_rate = side_outcome_counts["proposition"]["win"] / (side_outcome_counts["proposition"]["win"] + side_outcome_counts["proposition"]["loss"]) * 100
    opp_win_rate = side_outcome_counts["opposition"]["win"] / (side_outcome_counts["opposition"]["win"] + side_outcome_counts["opposition"]["loss"]) * 100

    results["side_advantage_test"] = {
        "proposition_win_rate": prop_win_rate,
        "opposition_win_rate": opp_win_rate,
        "chi_square": chi2,
        "p_value": chi2_p_value,
        "significant": chi2_p_value < 0.05,
        "interpretation": "Statistically significant side advantage" if chi2_p_value < 0.05 else "No significant side advantage"
    }

    # Fisher's exact test for small sample sizes
    oddsratio, fisher_p_value = fisher_exact(contingency_table)

    results["fisher_exact_test"] = {
        "odds_ratio": oddsratio,
        "p_value": fisher_p_value,
        "significant": fisher_p_value < 0.05,
        "interpretation": "Statistically significant side advantage (Fisher's exact test)"
                          if fisher_p_value < 0.05 else
                          "No significant side advantage (Fisher's exact test)"
    }

    # Test 4: Confidence difference by side
    if prop_opening_confidences and opp_opening_confidences:
        t_stat, p_value = ttest_ind(prop_opening_confidences, opp_opening_confidences)
        one_tailed_p = p_value / 2 if np.mean(prop_opening_confidences) > np.mean(opp_opening_confidences) else 1 - (p_value / 2)

        results["side_confidence_difference"] = {
            "proposition_mean": np.mean(prop_opening_confidences),
            "opposition_mean": np.mean(opp_opening_confidences),
            "difference": np.mean(prop_opening_confidences) - np.mean(opp_opening_confidences),
            "t_statistic": t_stat,
            "p_value": p_value,
            "one_tailed_p": one_tailed_p,
            "significant": one_tailed_p < 0.05,
            "interpretation": "Proposition has significantly higher confidence" if one_tailed_p < 0.05 and t_stat > 0 else
                             "Opposition has significantly higher confidence" if one_tailed_p < 0.05 and t_stat < 0 else
                             "No significant difference in confidence by side"
        }

        # Mann-Whitney U test (non-parametric alternative to independent t-test)
        u_stat, mw_p_value = mannwhitneyu(prop_opening_confidences, opp_opening_confidences, alternative='two-sided')
        mw_one_tailed_p = mw_p_value / 2 if np.mean(prop_opening_confidences) > np.mean(opp_opening_confidences) else 1 - (mw_p_value / 2)

        results["mannwhitney_side_difference"] = {
            "u_statistic": u_stat,
            "p_value": mw_p_value,
            "one_tailed_p": mw_one_tailed_p,
            "significant": mw_p_value < 0.05,
            "interpretation": "Significant difference in confidence distributions (Mann-Whitney U)"
                              if mw_p_value < 0.05 else
                              "No significant difference in confidence distributions (Mann-Whitney U)"
        }

    # Wilcoxon signed-rank test for overconfidence (non-parametric alternative to paired t-test)
    try:
        confidence_outcome_diff = np.array(all_opening_confidences) - np.array([x*100 for x in all_win_outcomes])
        w_stat, w_p_value = wilcoxon(confidence_outcome_diff)
        w_one_tailed_p = w_p_value / 2 if np.mean(confidence_outcome_diff) > 0 else 1 - (w_p_value / 2)

        results["wilcoxon_overconfidence_test"] = {
            "w_statistic": w_stat,
            "p_value": w_p_value,
            "one_tailed_p": w_one_tailed_p,
            "significant": w_p_value < 0.05,
            "interpretation": "Statistically significant overconfidence (Wilcoxon)"
                              if w_p_value < 0.05 and np.mean(confidence_outcome_diff) > 0 else
                              "Statistically significant underconfidence (Wilcoxon)"
                              if w_p_value < 0.05 and np.mean(confidence_outcome_diff) < 0 else
                              "No significant confidence bias (Wilcoxon)"
        }
    except Exception as e:
        results["wilcoxon_overconfidence_test"] = {
            "error": f"Failed to run Wilcoxon test: {str(e)}"
        }

    # Test 5: ANCOVA-like analysis using regression
    # Combine data for regression
    sides = [0] * len(prop_opening_confidences) + [1] * len(opp_opening_confidences)  # 0 for prop, 1 for opp
    confidences = prop_opening_confidences + opp_opening_confidences
    outcomes = prop_win_outcomes + opp_win_outcomes

    # Add constant term for intercept
    X = sm.add_constant(np.column_stack((sides, confidences)))
    model = sm.OLS(outcomes, X)
    results_sm = model.fit()

    results["regression_analysis"] = {
        "r_squared": results_sm.rsquared,
        "coefficients": {
            "intercept": {
                "value": results_sm.params[0],
                "p_value": results_sm.pvalues[0],
                "significant": results_sm.pvalues[0] < 0.05
            },
            "side": {
                "value": results_sm.params[1],
                "p_value": results_sm.pvalues[1],
                "significant": results_sm.pvalues[1] < 0.05
            },
            "confidence": {
                "value": results_sm.params[2],
                "p_value": results_sm.pvalues[2],
                "significant": results_sm.pvalues[2] < 0.05
            }
        },
        "interpretation": {
            "side_effect": "has" if results_sm.pvalues[1] < 0.05 else "does not have",
            "confidence_effect": "is" if results_sm.pvalues[2] < 0.05 else "is not",
            "side_advantage": "Opposition has an advantage"
                             if results_sm.pvalues[1] < 0.05 and results_sm.params[1] > 0 else
                             "Proposition has an advantage"
                             if results_sm.pvalues[1] < 0.05 and results_sm.params[1] < 0 else
                             "No side advantage detected"
        }
    }

    # Test 6: Model-specific overconfidence
    model_overconfidence_tests = {}

    # Group by model
    model_data = defaultdict(lambda: {"confidences": [], "outcomes": []})

    for debate in data["debates"]:
        # Process proposition
        if "opening" in debate.prop_bets:
            model = debate.proposition_model
            conf = debate.prop_bets["opening"]
            won = 1 if debate.winner == "proposition" else 0

            model_data[model]["confidences"].append(conf)
            model_data[model]["outcomes"].append(won)

        # Process opposition
        if "opening" in debate.opp_bets:
            model = debate.opposition_model
            conf = debate.opp_bets["opening"]
            won = 1 if debate.winner == "opposition" else 0

            model_data[model]["confidences"].append(conf)
            model_data[model]["outcomes"].append(won)

    # Run t-test for each model with enough samples
    for model, model_points in model_data.items():
        if len(model_points["confidences"]) >= 5:
            confs = model_points["confidences"]
            outcomes = [x*100 for x in model_points["outcomes"]]

            t_stat, p_value = ttest_rel(confs, outcomes)
            one_tailed_p = p_value / 2 if np.mean(confs) > np.mean(outcomes) else 1 - (p_value / 2)
            mean_diff = np.mean(np.array(confs) - np.array(outcomes))

            model_overconfidence_tests[model] = {
                "sample_size": len(confs),
                "mean_confidence": np.mean(confs),
                "win_rate": np.mean(model_points["outcomes"])*100,
                "mean_difference": mean_diff,
                "t_statistic": t_stat,
                "p_value": p_value,
                "one_tailed_p": one_tailed_p,
                "significant": one_tailed_p < 0.05,
                "interpretation": "OVERCONFIDENT" if one_tailed_p < 0.05 and mean_diff > 0 else
                                "UNDERCONFIDENT" if one_tailed_p < 0.05 and mean_diff < 0 else
                                "Well-calibrated"
            }

            # Add Wilcoxon test for models if sufficient data
            try:
                model_diff = np.array(confs) - np.array(outcomes)
                w_stat, w_p_value = wilcoxon(model_diff)
                w_one_tailed_p = w_p_value / 2 if np.mean(model_diff) > 0 else 1 - (w_p_value / 2)

                model_overconfidence_tests[model]["wilcoxon"] = {
                    "w_statistic": w_stat,
                    "p_value": w_p_value,
                    "one_tailed_p": w_one_tailed_p,
                    "significant": w_p_value < 0.05,
                    "interpretation": "OVERCONFIDENT (Wilcoxon)" if w_p_value < 0.05 and np.mean(model_diff) > 0 else
                                    "UNDERCONFIDENT (Wilcoxon)" if w_p_value < 0.05 and np.mean(model_diff) < 0 else
                                    "Well-calibrated (Wilcoxon)"
                }
            except Exception as e:
                model_overconfidence_tests[model]["wilcoxon"] = {
                    "error": f"Failed to run Wilcoxon test: {str(e)}"
                }

    results["model_overconfidence_tests"] = model_overconfidence_tests

    return results

def analyze_confidence_distribution(data: Dict[str, Any]) -> Dict[str, int]:
    """
    Analyze how many debates have both participants betting within specific confidence ranges
    based on their final (closing) bets.

    Args:
        data: The tournament data dictionary from load_tournament_data

    Returns:
        Dictionary with counts of debates in different confidence categories
    """
    # Initialize counters
    both_under_50 = 0
    both_51_to_74 = 0
    both_75_or_more = 0
    mixed_confidence = 0
    total_with_both_bets = 0

    # Process each debate
    for debate in data["debates"]:
        if "closing" in debate.prop_bets and "closing" in debate.opp_bets:
            total_with_both_bets += 1
            prop_conf = debate.prop_bets["closing"]
            opp_conf = debate.opp_bets["closing"]

            if prop_conf < 50 and opp_conf < 50:
                both_under_50 += 1
            elif (51 <= prop_conf <= 74) and (51 <= opp_conf <= 74):
                both_51_to_74 += 1
            elif prop_conf >= 75 and opp_conf >= 75:
                both_75_or_more += 1
            else:
                mixed_confidence += 1

    # Calculate percentages
    results = {
        "both_under_50": both_under_50,
        "both_51_to_74": both_51_to_74,
        "both_75_or_more": both_75_or_more,
        "mixed_confidence": mixed_confidence,
        "total_with_both_bets": total_with_both_bets
    }

    # Add percentages if we have data
    if total_with_both_bets > 0:
        results["both_under_50_pct"] = (both_under_50 / total_with_both_bets) * 100
        results["both_51_to_74_pct"] = (both_51_to_74 / total_with_both_bets) * 100
        results["both_75_or_more_pct"] = (both_75_or_more / total_with_both_bets) * 100
        results["mixed_confidence_pct"] = (mixed_confidence / total_with_both_bets) * 100
    return results

def analyze_model_betting_behavior(tournament_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Analyze betting behavior of each model across different speech types.

    Args:
        tournament_data: Dictionary containing tournament data, as returned by load_tournament_data()

    Returns:
        A dictionary with model names as keys, each containing a nested dictionary of:
            - speech_type (opening, rebuttal, closing): average bet amount
            - 'total_debates': number of debates the model participated in
    """
    # Initialize data structure to track betting behavior
    model_betting = {}

    # Initialize counters for each model's participation
    model_debate_counts = {}

    # Process all debates
    for debate in tournament_data["debates"]:
        # Track proposition model bets
        prop_model = debate.proposition_model
        if prop_model not in model_betting:
            model_betting[prop_model] = {
                "opening": 0.0,
                "rebuttal": 0.0,
                "closing": 0.0,
                "total_debates": 0
            }
            model_debate_counts[prop_model] = 0

        # Add proposition bets for each speech type
        for speech_type, bet_amount in debate.prop_bets.items():
            model_betting[prop_model][speech_type] += bet_amount

        model_debate_counts[prop_model] += 1

        # Track opposition model bets
        opp_model = debate.opposition_model
        if opp_model not in model_betting:
            model_betting[opp_model] = {
                "opening": 0.0,
                "rebuttal": 0.0,
                "closing": 0.0,
                "total_debates": 0
            }
            model_debate_counts[opp_model] = 0

        # Add opposition bets for each speech type
        for speech_type, bet_amount in debate.opp_bets.items():
            model_betting[opp_model][speech_type] += bet_amount

        model_debate_counts[opp_model] += 1

    # Calculate averages and update total debate counts
    for model in model_betting:
        model_betting[model]["total_debates"] = model_debate_counts[model]

        # Calculate average bet per speech type
        for speech_type in ["opening", "rebuttal", "closing"]:
            if model_debate_counts[model] > 0:
                model_betting[model][speech_type] /= model_debate_counts[model]

    return model_betting


def main():
    """
    Main function to run the complete tournament analysis.
    """
    import logging
    from pathlib import Path

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("tournament_analysis")

    # Set the tournament directories
    tournament_dirs = [
        Path("tournament/bet_tournament_20250316_1548"),
        Path("tournament/bet_tournament_20250317_1059")
    ]

    # Ask user which models to exclude
    print("\n=== AVAILABLE MODELS ===")
    # First load just the tournament results to get model names
    temp_data = load_tournament_data(tournament_dirs)
    all_models = list(temp_data["models"])

    for i, model in enumerate(all_models):
        print(f"[{i}] {model}")

    excluded_models = []
    exclude_option = input("\nDo you want to exclude any models from analysis? (y/n): ").strip().lower()

    if exclude_option == 'y':
        exclude_indices = input("Enter indices of models to exclude (comma-separated): ").strip()
        try:
            indices = [int(idx.strip()) for idx in exclude_indices.split(',') if idx.strip()]
            for idx in indices:
                if 0 <= idx < len(all_models):
                    excluded_model = all_models[idx]
                    excluded_models.append(excluded_model)
                    print(f"Excluding: {excluded_model}")
                else:
                    print(f"Warning: Invalid index {idx}, ignoring")
        except ValueError:
            print("Warning: Invalid input format. No models will be excluded.")

    # Load tournament data
    logger.info("Loading tournament data...")
    data = load_tournament_data(tournament_dirs, excluded_models)
    logger.info(f"Loaded {len(data['debates'])} debates with {len(data['models'])} models")

    # Run analyses
    logger.info("Running win rate analysis...")
    win_rates = analyze_win_rates_by_confidence(data)

    logger.info("Running confidence change analysis...")
    confidence_changes = analyze_model_confidence_changes(data)

    logger.info("Running judge agreement analysis...")
    judge_agreement = analyze_judge_agreement(data)

    logger.info("Running topic difficulty analysis...")
    topic_difficulty = analyze_topic_difficulty(data)

    logger.info("Running model calibration analysis...")
    model_calibration = analyze_model_calibration(data)

    logger.info("Running confidence gap analysis...")
    confidence_gaps = analyze_confidence_gaps(data)

    logger.info("Running overconfidence analysis...")
    overconfidence = analyze_overconfidence_metrics(data)

    logger.info("Running statistical tests...")
    statistical_tests = run_statistical_tests(data)

    # Print results
    print("\n" + "="*80)
    print("TOURNAMENT ANALYSIS RESULTS")
    print("="*80)

    # 1. Win rates by confidence level
    print("\n=== WIN RATES BY CONFIDENCE LEVEL ===")
    for tier, statistics in win_rates.items():
        print(f"Confidence {tier}: {statistics['wins']}/{statistics['total']} wins ({statistics['percentage']:.1f}%)")

    # 2. Average confidence change
    print("\n=== AVERAGE CONFIDENCE CHANGE ===")
    for model, statistics in confidence_changes.items():
        print(f"{model}: Avg change {statistics['avg_change']:.2f}, Avg absolute change {statistics['avg_abs_change']:.2f}")

    # 3. Judge agreement
    print("\n=== JUDGE AGREEMENT ===")
    print(f"Unanimous decisions: {judge_agreement['overall']['unanimous']}/{judge_agreement['overall']['total']} "
          f"({judge_agreement['overall']['unanimous_percent']:.1f}%)")
    print(f"Split decisions: {judge_agreement['overall']['split']}/{judge_agreement['overall']['total']} "
          f"({judge_agreement['overall']['split_percent']:.1f}%)")

    print("\nJudge decision distribution by number of dissenting judges:")
    for dissent_count, debate_count in sorted(judge_agreement['dissent_distribution'].items()):
        percentage = (debate_count / judge_agreement['overall']['total']) * 100
        print(f"  {dissent_count} dissenting judge{'s' if dissent_count != 1 else ''}: "
              f"{debate_count} debates ({percentage:.1f}%)")

    # 4. Topic difficulty index
    print("\n=== TOPIC DIFFICULTY INDEX ===")
    # Sort topics by difficulty index
    sorted_topics = sorted(topic_difficulty.items(),
                          key=lambda x: x[1]['difficulty_index'],
                          reverse=True)

    for topic, statistic in sorted_topics:
        print(f"{topic}: Difficulty index {statistic['difficulty_index']:.2f} "
              f"(Avg conf change {statistic['avg_confidence_change']:.2f}, "
              f"Judge disagreement {statistic['judge_disagreement_percent']:.1f}%)")

    # 5. Model calibration score
    print("\n=== MODEL CALIBRATION SCORE ===")
    # Sort models by calibration score (lower is better)
    sorted_models = sorted(model_calibration.items(),
                          key=lambda x: x[1]['calibration_score'])

    for model, statistic in sorted_models:
        print(f"{model}: Calibration score {statistic['calibration_score']:.4f}, "
              f"Confidence {statistic['avg_confidence']:.1f}%, Win rate {statistic['win_rate']:.1f}%")

    # 6. Confidence gap analysis
    print("\n=== CONFIDENCE GAP ANALYSIS ===")
    print(f"Average confidence gap: {confidence_gaps['overall']['avg_gap']:.2f}")
    print(f"Higher confidence side win rate: {confidence_gaps['overall']['higher_conf_win_rate']:.1f}%")

    for gap_type, statistic in confidence_gaps['by_gap_size'].items():
        print(f"{gap_type} gaps: {statistic['count']} debates, "
              f"avg gap {statistic['avg_gap']:.2f}, "
              f"higher conf win rate {statistic['higher_conf_win_rate']:.1f}%")

    # 7. Overconfidence metric
    print("\n=== OVERCONFIDENCE METRIC ===")
    for model, statistic in overconfidence.items():
        if statistic['high_conf_debates'] > 0:
            print(f"{model}: Lost {statistic['high_conf_losses']}/{statistic['high_conf_debates']} "
                  f"high-confidence debates ({statistic['high_conf_loss_rate']:.1f}%)")

    # 8. Statistical tests
    print("\n=== STATISTICAL HYPOTHESIS TESTING ===")

    # General overconfidence
    print("\n--- Hypothesis 1: General Overconfidence ---")
    t_test = statistical_tests['overconfidence_t_test']
    print("One-sample t-test (Confidence > 50):")
    print(f"  Mean confidence: {t_test['mean_confidence']:.2f}")
    print(f"  t-statistic: {t_test['t_statistic']:.3f}")
    print(f"  p-value: {t_test['one_tailed_p']:.4f} (one-tailed)")
    print(f"  Result: {t_test['interpretation']}")

    paired_test = statistical_tests['confidence_vs_outcome_t_test']
    print("\nPaired t-test (Confidence vs. Actual Win Rate):")
    print(f"  Mean confidence: {paired_test['mean_confidence']:.2f}")
    print(f"  Mean win rate: {paired_test['mean_win_rate']:.2f}%")
    print(f"  Mean difference: {paired_test['mean_difference']:.2f}")
    print(f"  t-statistic: {paired_test['t_statistic']:.3f}")
    print(f"  p-value: {paired_test['one_tailed_p']:.4f} (one-tailed)")
    print(f"  Result: {paired_test['interpretation']}")

    if 'wilcoxon_overconfidence_test' in statistical_tests and 'error' not in statistical_tests['wilcoxon_overconfidence_test']:
        wilcoxon_test = statistical_tests['wilcoxon_overconfidence_test']
        print("\nWilcoxon signed-rank test (non-parametric alternative):")
        print(f"  W-statistic: {wilcoxon_test['w_statistic']}")
        print(f"  p-value: {wilcoxon_test['one_tailed_p']:.4f} (one-tailed)")
        print(f"  Result: {wilcoxon_test['interpretation']}")

    # Side advantage
    print("\n--- Hypothesis 2: Proposition Disadvantage ---")
    side_test = statistical_tests['side_advantage_test']
    print("Chi-square test (Side vs. Outcome):")
    print(f"  Proposition: Win rate {side_test['proposition_win_rate']:.1f}%")
    print(f"  Opposition: Win rate {side_test['opposition_win_rate']:.1f}%")
    print(f"  Chi-square: {side_test['chi_square']:.3f}")
    print(f"  p-value: {side_test['p_value']:.4f}")
    print(f"  Result: {side_test['interpretation']}")

    fisher_test = statistical_tests['fisher_exact_test']
    print("\nFisher's exact test (for small samples):")
    print(f"  Odds ratio: {fisher_test['odds_ratio']:.3f}")
    print(f"  p-value: {fisher_test['p_value']:.4f}")
    print(f"  Result: {fisher_test['interpretation']}")

    if 'side_confidence_difference' in statistical_tests:
        conf_diff = statistical_tests['side_confidence_difference']
        print("\nIndependent t-test (Proposition vs. Opposition Confidence):")
        print(f"  Mean proposition confidence: {conf_diff['proposition_mean']:.2f}")
        print(f"  Mean opposition confidence: {conf_diff['opposition_mean']:.2f}")
        print(f"  Difference: {conf_diff['difference']:.2f}")
        print(f"  t-statistic: {conf_diff['t_statistic']:.3f}")
        print(f"  p-value: {conf_diff['one_tailed_p']:.4f} (one-tailed)")
        print(f"  Result: {conf_diff['interpretation']}")

    if 'mannwhitney_side_difference' in statistical_tests:
        mw_test = statistical_tests['mannwhitney_side_difference']
        print("\nMann-Whitney U test (non-parametric alternative):")
        print(f"  U-statistic: {mw_test['u_statistic']}")
        print(f"  p-value: {mw_test['p_value']:.4f}")
        print(f"  Result: {mw_test['interpretation']}")

    # Regression analysis
    if 'regression_analysis' in statistical_tests:
        reg = statistical_tests['regression_analysis']
        print("\nRegression analysis (ANCOVA-like approach):")
        print("  Model: Win ~ Intercept + Side + Confidence")
        print(f"  R-squared: {reg['r_squared']:.4f}")
        print("  Coefficients:")
        print(f"    Intercept: {reg['coefficients']['intercept']['value']:.4f} "
              f"(p={reg['coefficients']['intercept']['p_value']:.4f})")
        print(f"    Side (Opp=1): {reg['coefficients']['side']['value']:.4f} "
              f"(p={reg['coefficients']['side']['p_value']:.4f})")
        print(f"    Confidence: {reg['coefficients']['confidence']['value']:.4f} "
              f"(p={reg['coefficients']['confidence']['p_value']:.4f})")
        print(f"  Interpretation: After controlling for confidence, debate side "
              f"{reg['interpretation']['side_effect']} a significant effect on winning.")
        print(f"                  Confidence {reg['interpretation']['confidence_effect']} "
              f"a significant predictor of winning.")
        print(f"  {reg['interpretation']['side_advantage']}")

    # Model-specific overconfidence
    print("\n--- Model-Specific Overconfidence Analysis ---")
    print("\nOverconfidence by model (t-test comparing confidence to win rate):")

    for model, test_results in statistical_tests['model_overconfidence_tests'].items():
        print(f"{model}:")
        print(f"  Samples: {test_results['sample_size']}")
        print(f"  Mean confidence: {test_results['mean_confidence']:.2f}%")
        print(f"  Win rate: {test_results['win_rate']:.2f}%")
        print(f"  Mean difference: {test_results['mean_difference']:.2f}%")
        print(f"  t-statistic: {test_results['t_statistic']:.3f}")
        print(f"  p-value: {test_results['one_tailed_p']:.4f} (one-tailed)")
        print(f"  Result: {test_results['interpretation']}")

        if 'wilcoxon' in test_results and 'error' not in test_results['wilcoxon']:
            print(f"  Wilcoxon result: {test_results['wilcoxon']['interpretation']}")
        print("")

    logger.info("Running confidence distribution analysis...")
    confidence_distribution = analyze_confidence_distribution(data)

    # Add this to the results printing section:
    print("\n=== CONFIDENCE DISTRIBUTION ANALYSIS ===")
    print(f"Total debates with both bets: {confidence_distribution['total_with_both_bets']}")
    print(f"Both debaters under 50%: {confidence_distribution['both_under_50']} "
        f"({confidence_distribution.get('both_under_50_pct', 0):.1f}%)")
    print(f"Both debaters 51-74%: {confidence_distribution['both_51_to_74']} "
        f"({confidence_distribution.get('both_51_to_74_pct', 0):.1f}%)")
    print(f"Both debaters 75% or more: {confidence_distribution['both_75_or_more']} "
        f"({confidence_distribution.get('both_75_or_more_pct', 0):.1f}%)")
    print(f"Mixed confidence levels: {confidence_distribution['mixed_confidence']} "
        f"({confidence_distribution.get('mixed_confidence_pct', 0):.1f}%)")

    print("MODEL BETTING BEHAVIOR SUMMARY")
    model_betting = analyze_model_betting_behavior(data)
    print("==============================")
    print(f"{'Model':<20} {'Opening':<10} {'Rebuttal':<10} {'Closing':<10} {'Total Debates':<15}")
    print("-" * 65)

    for model, data in sorted(model_betting.items()):
        print(f"{model:<20} {data['opening']:<10.2f} {data['rebuttal']:<10.2f} {data['closing']:<10.2f} {data['total_debates']:<15}")

    print("\n")



    print("\nQuantitative analysis complete!")

if __name__ == "__main__":
    main()
