#!/usr/bin/env python3
"""
Script to extract detailed quantitative metrics from debate tournament data.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict

from core.models import DebateTotal, Side, SpeechType
import numpy as np
import statsmodels.api as sm
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, chi2_contingency, fisher_exact, wilcoxon, mannwhitneyu


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("quantitative_analysis")

# Set the tournament directories
tournament_dirs = [
    Path("tournament/bet_tournament_20250316_1548"),
    Path("tournament/bet_tournament_20250317_1059")
]

# Initialize combined model stats
combined_model_stats = {}

# Load tournament results from all directories
for tournament_dir in tournament_dirs:
    tournament_results_path = tournament_dir / "tournament_results.json"
    try:
        with open(tournament_results_path, "r") as f:
            tournament_results = json.load(f)
            # Merge model stats
            combined_model_stats.update(tournament_results.get("model_stats", {}))
            logger.info(f"Loaded tournament results from: {tournament_dir}")
    except FileNotFoundError:
        logger.warning(f"Tournament results file not found: {tournament_results_path}")
    except json.JSONDecodeError:
        logger.error(f"Error parsing JSON from: {tournament_results_path}")

# Print all available models with indices
print("\n=== AVAILABLE MODELS ===")
all_models = list(combined_model_stats.keys())
for i, model in enumerate(all_models):
    print(f"[{i}] {model}")

# Ask user which models to exclude
EXCLUDED_MODELS = []
exclude_option = input("\nDo you want to exclude any models from analysis? (y/n): ").strip().lower()

if exclude_option == 'y':
    exclude_indices = input("Enter indices of models to exclude (comma-separated): ").strip()
    try:
        indices = [int(idx.strip()) for idx in exclude_indices.split(',') if idx.strip()]
        for idx in indices:
            if 0 <= idx < len(all_models):
                excluded_model = all_models[idx]
                EXCLUDED_MODELS.append(excluded_model)
                print(f"Excluding: {excluded_model}")
            else:
                print(f"Warning: Invalid index {idx}, ignoring")
    except ValueError:
        print("Warning: Invalid input format. No models will be excluded.")

# Remove excluded models from stats
original_model_count = len(combined_model_stats)
for excluded_model in EXCLUDED_MODELS:
    if excluded_model in combined_model_stats:
        logger.info(f"Excluding model from stats: {excluded_model}")
        del(combined_model_stats[excluded_model])

if EXCLUDED_MODELS:
    logger.info(f"Excluded {original_model_count - len(combined_model_stats)} models from analysis")

# Find all debate files from all tournament directories
all_debates = []

for tournament_dir in tournament_dirs:
    round_dirs = [d for d in tournament_dir.glob("round_*") if d.is_dir()]

    for round_dir in round_dirs:
        round_num = int(round_dir.name.split("_")[1])
        for debate_path in round_dir.glob("*.json"):
            try:
                debate = DebateTotal.load_from_json(debate_path)

                # Skip debates involving any excluded model
                if any(excluded_model in [debate.proposition_model, debate.opposition_model]
                       for excluded_model in EXCLUDED_MODELS):
                    logger.info(f"Skipping debate with excluded model: {debate_path.name}")
                    continue

                all_debates.append({
                    "round": round_num,
                    "path": debate_path,
                    "debate": debate,
                    "prop_model": debate.proposition_model,
                    "opp_model": debate.opposition_model,
                    "topic": debate.motion.topic_description,
                    "tournament": tournament_dir.name
                })
                logger.info(f"Loaded debate: {debate_path.name} from {tournament_dir.name}")
            except Exception as e:
                logger.error(f"Failed to load debate {debate_path}: {str(e)}")

logger.info(f"Loaded {len(all_debates)} debates across {len(tournament_dirs)} tournaments")

# Initialize data structures for quantitative analysis
win_rates_by_confidence = {
    "0-25": {"wins": 0, "total": 0},
    "26-50": {"wins": 0, "total": 0},
    "51-75": {"wins": 0, "total": 0},
    "76-100": {"wins": 0, "total": 0}
}

model_confidence_changes = defaultdict(list)
model_confidence_in_wins = defaultdict(list)
model_confidence_in_losses = defaultdict(list)
topic_difficulty = defaultdict(lambda: {"confidence_changes": [], "judge_disagreements": 0, "total_debates": 0})
model_calibration = defaultdict(lambda: {"confidence": [], "win": []})
confidence_gaps = []
round_confidence = defaultdict(lambda: {"opening": [], "rebuttal": [], "closing": []})
overconfidence_metrics = defaultdict(lambda: {"high_conf_losses": 0, "high_conf_debates": 0})
judge_agreement_count = {"unanimous": 0, "split": 0}

# Calibration data structures
calibration_opp_only = defaultdict(lambda: {"confidence": [], "win": []})
calibration_high_agreement = defaultdict(lambda: {"confidence": [], "win": []})
calibration_winning_models = defaultdict(lambda: {"confidence": [], "win": []})

# NEW: Additional Calibration Data Structures
calibration_prop_only = defaultdict(lambda: {"confidence": [], "win": []})
calibration_losing_models = defaultdict(lambda: {"confidence": [], "win": []})
calibration_low_agreement = defaultdict(lambda: {"confidence": [], "win": []})

# Extract quantitative data from debates
for debate_data in all_debates:
    debate = debate_data["debate"]
    if not debate.debator_bets or not debate.judge_results:
        continue

    # Determine winner from judge results
    winner_counts = {"proposition": 0, "opposition": 0}
    for result in debate.judge_results:
        winner_counts[result.winner] += 1

    winner = "proposition" if winner_counts["proposition"] > winner_counts["opposition"] else "opposition"

    # Check for judge agreement
    total_judges = len(debate.judge_results)
    max_agreement = max(winner_counts.values())
    is_high_agreement = max_agreement == total_judges  # Boolean for high agreement
    is_low_agreement = max_agreement < total_judges   # NEW: Boolean for low agreement

    if max_agreement == total_judges:
        judge_agreement_count["unanimous"] += 1
    else:
        judge_agreement_count["split"] += 1
        topic = debate.motion.topic_description
        topic_difficulty[topic]["judge_disagreements"] += 1

    topic_difficulty[topic]["total_debates"] += 1

    # Initialize variables to track bets by speech type
    prop_bets = {}
    opp_bets = {}

    # Extract all bets from the debate
    for bet in debate.debator_bets:
        if bet.side == Side.PROPOSITION:
            prop_bets[bet.speech_type] = bet.amount
        elif bet.side == Side.OPPOSITION:
            opp_bets[bet.speech_type] = bet.amount

    # Process opening and closing bets if available
    if SpeechType.OPENING in prop_bets and SpeechType.CLOSING in prop_bets:
        prop_change = prop_bets[SpeechType.CLOSING] - prop_bets[SpeechType.OPENING]
        model_confidence_changes[debate.proposition_model].append(prop_change)
        topic_difficulty[topic]["confidence_changes"].append(abs(prop_change))

        # Track round-by-round confidence
        for speech_type, bet in prop_bets.items():
            round_confidence[debate.proposition_model][speech_type.value].append(bet)

        # Track confidence by win/loss
        if winner == "proposition":
            model_confidence_in_wins[debate.proposition_model].append(prop_bets[SpeechType.OPENING])
        else:
            model_confidence_in_losses[debate.proposition_model].append(prop_bets[SpeechType.OPENING])

        # Track for win rates by confidence tier
        prop_initial = prop_bets[SpeechType.OPENING]
        tier = "0-25" if prop_initial <= 25 else "26-50" if prop_initial <= 50 else "51-75" if prop_initial <= 75 else "76-100"
        win_rates_by_confidence[tier]["total"] += 1
        if winner == "proposition":
            win_rates_by_confidence[tier]["wins"] += 1

        # Track for model calibration
        model_calibration[debate.proposition_model]["confidence"].append(prop_initial)
        model_calibration[debate.proposition_model]["win"].append(1 if winner == "proposition" else 0)

        # Track for overconfidence metric
        if prop_initial > 75:
            overconfidence_metrics[debate.proposition_model]["high_conf_debates"] += 1
            if winner != "proposition":
                overconfidence_metrics[debate.proposition_model]["high_conf_losses"] += 1

        # Calibration - Winning Models (Prop)
        if winner == "proposition":
            calibration_winning_models[debate.proposition_model]["confidence"].append(prop_initial)
            calibration_winning_models[debate.proposition_model]["win"].append(1)

        # Calibration - High Agreement (Prop)
        if is_high_agreement:
            calibration_high_agreement[debate.proposition_model]["confidence"].append(prop_initial)
            calibration_high_agreement[debate.proposition_model]["win"].append(1 if winner == "proposition" else 0)

        # NEW: Calibration - Proposition Models Only
        calibration_prop_only[debate.proposition_model]["confidence"].append(prop_initial)
        calibration_prop_only[debate.proposition_model]["win"].append(1 if winner == "proposition" else 0)

        # NEW: Calibration - Losing Models (Prop)
        if winner != "proposition":
            calibration_losing_models[debate.proposition_model]["confidence"].append(prop_initial)
            calibration_losing_models[debate.proposition_model]["win"].append(0)

        # NEW: Calibration - Low Agreement Debates (Prop)
        if is_low_agreement:
            calibration_low_agreement[debate.proposition_model]["confidence"].append(prop_initial)
            calibration_low_agreement[debate.proposition_model]["win"].append(1 if winner == "proposition" else 0)

    if SpeechType.OPENING in opp_bets and SpeechType.CLOSING in opp_bets:
        opp_change = opp_bets[SpeechType.CLOSING] - opp_bets[SpeechType.OPENING]
        model_confidence_changes[debate.opposition_model].append(opp_change)
        topic_difficulty[topic]["confidence_changes"].append(abs(opp_change))

        # Track round-by-round confidence
        for speech_type, bet in opp_bets.items():
            round_confidence[debate.opposition_model][speech_type.value].append(bet)

        # Track confidence by win/loss
        if winner == "opposition":
            model_confidence_in_wins[debate.opposition_model].append(opp_bets[SpeechType.OPENING])
        else:
            model_confidence_in_losses[debate.opposition_model].append(opp_bets[SpeechType.OPENING])

        # Track for win rates by confidence tier
        opp_initial = opp_bets[SpeechType.OPENING]
        tier = "0-25" if opp_initial <= 25 else "26-50" if opp_initial <= 50 else "51-75" if opp_initial <= 75 else "76-100"
        win_rates_by_confidence[tier]["total"] += 1
        if winner == "opposition":
            win_rates_by_confidence[tier]["wins"] += 1

        # Track for model calibration
        model_calibration[debate.opposition_model]["confidence"].append(opp_initial)
        model_calibration[debate.opposition_model]["win"].append(1 if winner == "opposition" else 0)

        # Track for overconfidence metric
        if opp_initial > 75:
            overconfidence_metrics[debate.opposition_model]["high_conf_debates"] += 1
            if winner != "opposition":
                overconfidence_metrics[debate.opposition_model]["high_conf_losses"] += 1

        # Calibration - Opposition Only
        calibration_opp_only[debate.opposition_model]["confidence"].append(opp_initial)
        calibration_opp_only[debate.opposition_model]["win"].append(1 if winner == "opposition" else 0)

        # Calibration - Winning Models (Opp)
        if winner == "opposition":
            calibration_winning_models[debate.opposition_model]["confidence"].append(opp_initial)
            calibration_winning_models[debate.opposition_model]["win"].append(1)

        # Calibration - High Agreement (Opp)
        if is_high_agreement:
            calibration_high_agreement[debate.opposition_model]["confidence"].append(opp_initial)
            calibration_high_agreement[debate.opposition_model]["win"].append(1 if winner == "opposition" else 0)

        # NEW: Calibration - Losing Models (Opp)
        if winner != "opposition":
            calibration_losing_models[debate.opposition_model]["confidence"].append(opp_initial)
            calibration_losing_models[debate.opposition_model]["win"].append(0)

        # NEW: Calibration - Low Agreement Debates (Opp)
        if is_low_agreement:
            calibration_low_agreement[debate.opposition_model]["confidence"].append(opp_initial)
            calibration_low_agreement[debate.opposition_model]["win"].append(1 if winner == "opposition" else 0)

    # Calculate confidence gap if both sides have opening bets
    if SpeechType.OPENING in prop_bets and SpeechType.OPENING in opp_bets:
        confidence_gap = abs(prop_bets[SpeechType.OPENING] - opp_bets[SpeechType.OPENING])
        confidence_gaps.append((confidence_gap, winner))

# Print quantitative findings

# 1. Win rates by confidence level
print("\n=== WIN RATES BY CONFIDENCE LEVEL ===")
for tier, data in win_rates_by_confidence.items():
    if data["total"] > 0:
        win_rate = (data["wins"] / data["total"]) * 100
        print(f"Confidence {tier}: {data['wins']}/{data['total']} wins ({win_rate:.1f}%)")

# 2. Average confidence change
print("\n=== AVERAGE CONFIDENCE CHANGE ===")
for model, changes in model_confidence_changes.items():
    if changes:
        avg_change = sum(changes) / len(changes)
        avg_abs_change = sum(abs(c) for c in changes) / len(changes)
        print(f"{model}: Avg change {avg_change:.2f}, Avg absolute change {avg_abs_change:.2f}")

# 3. Confidence accuracy ratio
print("\n=== CONFIDENCE ACCURACY RATIO ===")
for model in set(model_confidence_in_wins.keys()) | set(model_confidence_in_losses.keys()):
    wins = model_confidence_in_wins.get(model, [])
    losses = model_confidence_in_losses.get(model, [])

    if wins and losses:
        avg_win_conf = sum(wins) / len(wins)
        avg_loss_conf = sum(losses) / len(losses)
        ratio = avg_win_conf / avg_loss_conf if avg_loss_conf > 0 else float('inf')

        print(f"{model}: Avg confidence in wins {avg_win_conf:.2f}, in losses {avg_loss_conf:.2f}, ratio {ratio:.2f}")

# 4. Judge agreement
# 4. Judge agreement
print("\n=== JUDGE AGREEMENT ===")
total_judged = judge_agreement_count["unanimous"] + judge_agreement_count["split"]
if total_judged > 0:
    # Create a distribution by number of dissenting judges
    dissent_distribution = defaultdict(int)

    for debate_data in all_debates:
        debate = debate_data["debate"]
        if not debate.judge_results:
            continue

        winner_counts = {"proposition": 0, "opposition": 0}
        for result in debate.judge_results:
            winner_counts[result.winner] += 1

        total_judges = len(debate.judge_results)
        max_votes = max(winner_counts.values())
        dissenting_votes = total_judges - max_votes

        dissent_distribution[dissenting_votes] += 1

    # Display the unanimous and split statistics
    unanimous_pct = (judge_agreement_count["unanimous"] / total_judged) * 100
    split_pct = (judge_agreement_count["split"] / total_judged) * 100
    print(f"Unanimous decisions: {judge_agreement_count['unanimous']}/{total_judged} ({unanimous_pct:.1f}%)")
    print(f"Split decisions: {judge_agreement_count['split']}/{total_judged} ({split_pct:.1f}%)")

    # Display the detailed distribution by number of dissenting judges
    print("\nJudge decision distribution by number of dissenting judges:")
    for dissent_count, debate_count in sorted(dissent_distribution.items()):
        percentage = (debate_count / total_judged) * 100
        print(f"  {dissent_count} dissenting judge{'s' if dissent_count != 1 else ''}: {debate_count} debates ({percentage:.1f}%)")


# 5. Topic difficulty index
print("\n=== TOPIC DIFFICULTY INDEX ===")
for topic, data in topic_difficulty.items():
    if data["total_debates"] > 0:
        avg_conf_change = sum(data["confidence_changes"]) / len(data["confidence_changes"]) if data["confidence_changes"] else 0
        judge_disagreement_pct = (data["judge_disagreements"] / data["total_debates"]) * 100
        difficulty_index = avg_conf_change + judge_disagreement_pct

        print(f"{topic}: Difficulty index {difficulty_index:.2f} (Avg conf change {avg_conf_change:.2f}, Judge disagreement {judge_disagreement_pct:.1f}%)")

# 6. Model calibration score
print("\n=== MODEL CALIBRATION SCORE ===")
for model, data in model_calibration.items():
    if data["confidence"] and data["win"]:
        n = len(data["confidence"])
        # Calculate a simple calibration score (lower is better)
        calibration_score = sum((data["confidence"][i]/100 - data["win"][i])**2 for i in range(n)) / n
        print(f"{model}: Calibration score {calibration_score:.4f} (lower is better)")

# 7. Confidence gap analysis
print("\n=== CONFIDENCE GAP ANALYSIS ===")
if confidence_gaps:
    avg_gap = sum(gap for gap, _ in confidence_gaps) / len(confidence_gaps)

    # Group into small, medium, large gaps
    small_gaps = [(gap, winner) for gap, winner in confidence_gaps if gap <= 25]
    medium_gaps = [(gap, winner) for gap, winner in confidence_gaps if 25 < gap <= 50]
    large_gaps = [(gap, winner) for gap, winner in confidence_gaps if gap > 50]

    print(f"Average confidence gap: {avg_gap:.2f}")

    # Calculate how often the more confident side wins
    higher_confidence_wins = 0
    for gap, winner in confidence_gaps:
        prop_conf = next((gap[0] for gap in confidence_gaps if gap[1] == winner), 0)
        opp_conf = next((gap[0] for gap in confidence_gaps if gap[1] != winner), 0)
        if (winner == "proposition" and prop_conf > opp_conf) or (winner == "opposition" and opp_conf > prop_conf):
            higher_confidence_wins += 1

    higher_conf_win_rate = (higher_confidence_wins / len(confidence_gaps)) * 100 if confidence_gaps else 0
    print(f"Higher confidence side win rate: {higher_conf_win_rate:.1f}%")

    for gap_type, gaps in [("Small gaps (≤25)", small_gaps), ("Medium gaps (26-50)", medium_gaps), ("Large gaps (>50)", large_gaps)]:
        if gaps:
            print(f"{gap_type}: {len(gaps)} debates, avg gap {sum(gap for gap, _ in gaps)/len(gaps):.2f}")

# 8. Round-by-round confidence trends
print("\n=== ROUND-BY-ROUND CONFIDENCE TRENDS ===")
for model, data in round_confidence.items():
    if all(data.values()):  # Make sure we have data for all speech types
        avg_opening = sum(data["opening"]) / len(data["opening"])
        avg_rebuttal = sum(data["rebuttal"]) / len(data["rebuttal"]) if data["rebuttal"] else 0
        avg_closing = sum(data["closing"]) / len(data["closing"])

        print(f"{model}: Opening {avg_opening:.2f} → Rebuttal {avg_rebuttal:.2f} → Closing {avg_closing:.2f}")

# 9. Overconfidence metric
print("\n=== OVERCONFIDENCE METRIC ===")
for model, data in overconfidence_metrics.items():
    if data["high_conf_debates"] > 0:
        overconf_rate = (data["high_conf_losses"] / data["high_conf_debates"]) * 100
        print(f"{model}: Lost {data['high_conf_losses']}/{data['high_conf_debates']} high-confidence debates ({overconf_rate:.1f}%)")

# 10. Confidence vs Topic Contentiousness
print("\n=== CONFIDENCE VS TOPIC CONTENTIOUSNESS ===")

# Create a contentiousness score for each topic (0-100)
topic_contentiousness = {}
for topic, data in topic_difficulty.items():
    if data["total_debates"] > 0:
        # Use judge disagreement percentage as the contentiousness score
        contentiousness = (data["judge_disagreements"] / data["total_debates"]) * 100
        topic_contentiousness[topic] = {
            "contentiousness": contentiousness,
            "initial_confidences": [],
            "winner_confidences": [],
            "loser_confidences": []
        }

# Now collect confidence data for each topic
for debate_data in all_debates:
    debate = debate_data["debate"]
    if not debate.debator_bets or not debate.judge_results:
        continue

    topic = debate.motion.topic_description

    # Determine winner from judge results
    winner_counts = {"proposition": 0, "opposition": 0}
    for result in debate.judge_results:
        winner_counts[result.winner] += 1

    winner = "proposition" if winner_counts["proposition"] > winner_counts["opposition"] else "opposition"

    # Get initial confidence bets
    prop_initial = None
    opp_initial = None

    for bet in debate.debator_bets:
        if bet.speech_type == SpeechType.OPENING:
            if bet.side == Side.PROPOSITION:
                prop_initial = bet.amount
            elif bet.side == Side.OPPOSITION:
                opp_initial = bet.amount

    if prop_initial is not None and opp_initial is not None and topic in topic_contentiousness:
        # Add to general confidences
        topic_contentiousness[topic]["initial_confidences"].extend([prop_initial, opp_initial])

        # Add to winner/loser confidences
        if winner == "proposition":
            topic_contentiousness[topic]["winner_confidences"].append(prop_initial)
            topic_contentiousness[topic]["loser_confidences"].append(opp_initial)
        else:
            topic_contentiousness[topic]["winner_confidences"].append(opp_initial)
            topic_contentiousness[topic]["loser_confidences"].append(prop_initial)

# Print results sorted by contentiousness
print("\nTopic contentiousness and confidence patterns:")
print("Topic | Contentiousness | Avg Confidence | Winner Conf | Loser Conf | Loser/Winner Ratio")
print("-" * 100)

# Sort topics by contentiousness
for topic, data in sorted(topic_contentiousness.items(), key=lambda x: x[1]["contentiousness"], reverse=True):
    # Calculate average confidences
    avg_conf = sum(data["initial_confidences"]) / len(data["initial_confidences"]) if data["initial_confidences"] else 0
    winner_conf = sum(data["winner_confidences"]) / len(data["winner_confidences"]) if data["winner_confidences"] else 0
    loser_conf = sum(data["loser_confidences"]) / len(data["loser_confidences"]) if data["loser_confidences"] else 0

    # Calculate loser/winner ratio - this shows what % of confidence goes to the loser
    loser_winner_ratio = (loser_conf / winner_conf) * 100 if winner_conf > 0 else 0

    # Print in a table format with truncated topic
    topic_short = topic[:40] + "..." if len(topic) > 40 else topic.ljust(43)
    print(f"{topic_short} | {data['contentiousness']:14.1f}% | {avg_conf:14.2f} | {winner_conf:11.2f} | {loser_conf:10.2f} | {loser_winner_ratio:17.1f}%")

# Check if confidence correlates with contentiousness
contentiousness_values = []
confidence_values = []
loser_winner_ratios = []

for data in topic_contentiousness.values():
    if data["initial_confidences"] and data["winner_confidences"] and data["loser_confidences"]:
        contentiousness_values.append(data["contentiousness"])
        confidence_values.append(sum(data["initial_confidences"]) / len(data["initial_confidences"]))

        winner_conf = sum(data["winner_confidences"]) / len(data["winner_confidences"])
        loser_conf = sum(data["loser_confidences"]) / len(data["loser_confidences"])
        loser_winner_ratios.append((loser_conf / winner_conf) * 100 if winner_conf > 0 else 0)

# Simple correlation analysis
if len(contentiousness_values) >= 3:  # Need at least 3 points for meaningful correlation
    confidence_correlation = sum((contentiousness_values[i] - sum(contentiousness_values)/len(contentiousness_values)) *
                              (confidence_values[i] - sum(confidence_values)/len(confidence_values))
                              for i in range(len(contentiousness_values)))
    confidence_correlation /= (sum((v - sum(contentiousness_values)/len(contentiousness_values))**2 for v in contentiousness_values) *
                            sum((v - sum(confidence_values)/len(confidence_values))**2 for v in confidence_values))**0.5

    ratio_correlation = sum((contentiousness_values[i] - sum(contentiousness_values)/len(contentiousness_values)) *
                          (loser_winner_ratios[i] - sum(loser_winner_ratios)/len(loser_winner_ratios))
                          for i in range(len(contentiousness_values)))
    ratio_correlation /= (sum((v - sum(contentiousness_values)/len(contentiousness_values))**2 for v in contentiousness_values) *
                        sum((v - sum(loser_winner_ratios)/len(loser_winner_ratios))**2 for v in loser_winner_ratios))**0.5

    print("\nCorrelation between topic contentiousness and average confidence:", f"{confidence_correlation:.2f}")
    print("Correlation between topic contentiousness and loser/winner confidence ratio:", f"{ratio_correlation:.2f}")

    if confidence_correlation < 0:
        print("\nModels DO show lower confidence on more contentious topics, suggesting some ability to sense difficulty.")
    else:
        print("\nModels do NOT show lower confidence on more contentious topics, suggesting poor calibration to topic difficulty.")

    if ratio_correlation > 0:
        print("\nLoser/winner confidence ratio increases with topic contentiousness, suggesting models are less able to identify winning arguments on difficult topics.")

# Add tournament comparison section
print("\n=== TOURNAMENT COMPARISON ===")

# Group debates by tournament
debates_by_tournament = defaultdict(list)
for debate_data in all_debates:
    tournament = debate_data["tournament"]
    debates_by_tournament[tournament].append(debate_data)

# Compare key metrics across tournaments
for tournament, debates in debates_by_tournament.items():
    print(f"\n{tournament} - {len(debates)} debates")

    # Calculate tournament-specific judge agreement
    tournament_judge_agreement = {"unanimous": 0, "split": 0}

    for debate_data in debates:
        debate = debate_data["debate"]
        if not debate.judge_results:
            continue

        winner_counts = {"proposition": 0, "opposition": 0}
        for result in debate.judge_results:
            winner_counts[result.winner] += 1

        total_judges = len(debate.judge_results)
        max_agreement = max(winner_counts.values())
        if max_agreement == total_judges:
            tournament_judge_agreement["unanimous"] += 1
        else:
            tournament_judge_agreement["split"] += 1

    total_judged = tournament_judge_agreement["unanimous"] + tournament_judge_agreement["split"]
    if total_judged > 0:
        unanimous_pct = (tournament_judge_agreement["unanimous"] / total_judged) * 100
        split_pct = (tournament_judge_agreement["split"] / total_judged) * 100
        print(f"  Judge Agreement: {tournament_judge_agreement['unanimous']}/{total_judged} unanimous ({unanimous_pct:.1f}%)")

    # Calculate average confidence
    tournament_confidences = []
    for debate_data in debates:
        debate = debate_data["debate"]
        for bet in debate.debator_bets:
            if bet.speech_type == SpeechType.OPENING:
                tournament_confidences.append(bet.amount)

    if tournament_confidences:
        avg_conf = sum(tournament_confidences) / len(tournament_confidences)
        print(f"  Average opening confidence: {avg_conf:.2f}")

# Calibration Breakdowns
print("\n=== CALIBRATION BREAKDOWN ===")

# 1. Opposition Models Only
print("\n--- Calibration: Opposition Models Only ---")
for model, data in calibration_opp_only.items():
    if data["confidence"] and data["win"]:
        n = len(data["confidence"])
        calibration_score = sum((data["confidence"][i]/100 - data["win"][i])**2 for i in range(n)) / n
        print(f"{model}: Calibration score {calibration_score:.4f} (lower is better)")

# 2. High-Agreement Debates Only
print("\n--- Calibration: High-Agreement Debates Only ---")
for model, data in calibration_high_agreement.items():
    if data["confidence"] and data["win"]:
        n = len(data["confidence"])
        calibration_score = sum((data["confidence"][i]/100 - data["win"][i])**2 for i in range(n)) / n
        print(f"{model}: Calibration score {calibration_score:.4f} (lower is better)")

# 3. Winning Models Only
print("\n--- Calibration: Winning Models Only ---")
for model, data in calibration_winning_models.items():
    if data["confidence"] and data["win"]:
        n = len(data["confidence"])
        calibration_score = sum((data["confidence"][i]/100 - data["win"][i])**2 for i in range(n)) / n
        print(f"{model}: Calibration score {calibration_score:.4f} (lower is better)")

# NEW: 4. Proposition Models Only
print("\n--- Calibration: Proposition Models Only ---")
for model, data in calibration_prop_only.items():
    if data["confidence"] and data["win"]:
        n = len(data["confidence"])
        calibration_score = sum((data["confidence"][i]/100 - data["win"][i])**2 for i in range(n)) / n
        print(f"{model}: Calibration score {calibration_score:.4f} (lower is better)")

# NEW: 5. Losing Models Only
print("\n--- Calibration: Losing Models Only ---")
for model, data in calibration_losing_models.items():
    if data["confidence"] and data["win"]:
        n = len(data["confidence"])
        calibration_score = sum((data["confidence"][i]/100 - data["win"][i])**2 for i in range(n)) / n
        print(f"{model}: Calibration score {calibration_score:.4f} (lower is better)")

# NEW: 6. Low-Agreement Debates Only
print("\n--- Calibration: Low-Agreement Debates Only ---")
for model, data in calibration_low_agreement.items():
    if data["confidence"] and data["win"]:
        n = len(data["confidence"])
        calibration_score = sum((data["confidence"][i]/100 - data["win"][i])**2 for i in range(n)) / n
        print(f"{model}: Calibration score {calibration_score:.4f} (lower is better)")

print("\nQuantitative analysis complete!")

# === STATISTICAL HYPOTHESIS TESTING ===
print("\n=== STATISTICAL HYPOTHESIS TESTING ===")

# Prepare data structures for statistical tests
all_opening_confidences = []
all_win_outcomes = []
prop_opening_confidences = []
prop_win_outcomes = []
opp_opening_confidences = []
opp_win_outcomes = []

# Contingency table for side vs. outcome
side_outcome_counts = {"proposition": {"win": 0, "loss": 0},
                       "opposition": {"win": 0, "loss": 0}}

for debate_data in all_debates:
    debate = debate_data["debate"]
    if not debate.debator_bets or not debate.judge_results:
        continue

    # Determine winner
    winner_counts = {"proposition": 0, "opposition": 0}
    for result in debate.judge_results:
        winner_counts[result.winner] += 1

    winner = "proposition" if winner_counts["proposition"] > winner_counts["opposition"] else "opposition"

    # Extract opening confidences
    prop_conf = None
    opp_conf = None

    for bet in debate.debator_bets:
        if bet.speech_type == SpeechType.OPENING:
            if bet.side == Side.PROPOSITION:
                prop_conf = bet.amount
            elif bet.side == Side.OPPOSITION:
                opp_conf = bet.amount

    # Update data structures if we have both confidences
    if prop_conf is not None and opp_conf is not None:
        # Update proposition data
        prop_opening_confidences.append(prop_conf)
        prop_win = 1 if winner == "proposition" else 0
        prop_win_outcomes.append(prop_win)

        # Update opposition data
        opp_opening_confidences.append(opp_conf)
        opp_win = 1 if winner == "opposition" else 0
        opp_win_outcomes.append(opp_win)

        # Update all data
        all_opening_confidences.extend([prop_conf, opp_conf])
        all_win_outcomes.extend([prop_win, opp_win])

        # Update contingency table
        if winner == "proposition":
            side_outcome_counts["proposition"]["win"] += 1
            side_outcome_counts["opposition"]["loss"] += 1
        else:
            side_outcome_counts["proposition"]["loss"] += 1
            side_outcome_counts["opposition"]["win"] += 1

# Create contingency table for chi-square test
contingency_table = [
    [side_outcome_counts["proposition"]["win"], side_outcome_counts["proposition"]["loss"]],
    [side_outcome_counts["opposition"]["win"], side_outcome_counts["opposition"]["loss"]]
]

print("\n--- Hypothesis 1: General Overconfidence ---")

# Test 1: One-sample t-test for overall confidence > 50
t_stat, p_value = ttest_1samp(all_opening_confidences, 50)
print("One-sample t-test (Confidence > 50):")
print(f"  Mean confidence: {np.mean(all_opening_confidences):.2f}")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value/2:.4f} (one-tailed)")  # Divide by 2 for one-tailed test
print(f"  Result: {'Statistically significant overconfidence' if p_value/2 < 0.05 else 'No significant overconfidence'}")

# Test 2: Paired t-test for confidence vs. outcomes
paired_t_stat, paired_p_value = ttest_rel(all_opening_confidences, [x*100 for x in all_win_outcomes])
print("\nPaired t-test (Confidence vs. Actual Win Rate):")
print(f"  Mean confidence: {np.mean(all_opening_confidences):.2f}")
print(f"  Mean win rate: {np.mean(all_win_outcomes)*100:.2f}%")
print(f"  Mean difference: {np.mean(np.array(all_opening_confidences) - np.array(all_win_outcomes)*100):.2f}")
print(f"  t-statistic: {paired_t_stat:.3f}")
print(f"  p-value: {paired_p_value/2:.4f} (one-tailed)")  # Divide by 2 for one-tailed test
print(f"  Result: {'Statistically significant overconfidence' if paired_p_value/2 < 0.05 else 'No significant overconfidence'}")

# Non-parametric alternative: Wilcoxon signed-rank test
if len(all_opening_confidences) > 20:  # Only run if we have enough samples
    try:
        w_stat, w_p_value = wilcoxon(np.array(all_opening_confidences) - np.array(all_win_outcomes)*100)
        print("\nWilcoxon signed-rank test (non-parametric alternative):")
        print(f"  W-statistic: {w_stat}")
        print(f"  p-value: {w_p_value/2:.4f} (one-tailed)")
        print(f"  Result: {'Statistically significant overconfidence' if w_p_value/2 < 0.05 else 'No significant overconfidence'}")
    except Exception as e:
        print(f"\nWilcoxon test failed: {str(e)}")

print("\n--- Hypothesis 2: Proposition Disadvantage and Metacognitive Failure ---")

# Test 1: Chi-square test for side vs. outcome
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print("Chi-square test (Side vs. Outcome):")
print("  Contingency table:")
print(f"    Proposition: {contingency_table[0][0]} wins, {contingency_table[0][1]} losses ({contingency_table[0][0]/(contingency_table[0][0]+contingency_table[0][1])*100:.1f}% win rate)")
print(f"    Opposition: {contingency_table[1][0]} wins, {contingency_table[1][1]} losses ({contingency_table[1][0]/(contingency_table[1][0]+contingency_table[1][1])*100:.1f}% win rate)")
print(f"  Chi-square: {chi2:.3f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Result: {'Statistically significant proposition disadvantage' if p_value < 0.05 else 'No significant proposition disadvantage'}")

# Fisher's exact test for small sample sizes
oddsratio, fisher_p_value = fisher_exact(contingency_table)
print("\nFisher's exact test (for small samples):")
print(f"  Odds ratio: {oddsratio:.3f}")
print(f"  p-value: {fisher_p_value:.4f}")
print(f"  Result: {'Statistically significant proposition disadvantage' if fisher_p_value < 0.05 else 'No significant proposition disadvantage'}")

# Test 2: Independent t-test for proposition vs opposition confidence
t_stat, p_value = ttest_ind(prop_opening_confidences, opp_opening_confidences)
print("\nIndependent t-test (Proposition vs. Opposition Confidence):")
print(f"  Mean proposition confidence: {np.mean(prop_opening_confidences):.2f}")
print(f"  Mean opposition confidence: {np.mean(opp_opening_confidences):.2f}")
print(f"  Difference: {np.mean(prop_opening_confidences) - np.mean(opp_opening_confidences):.2f}")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value/2:.4f} (one-tailed for prop > opp)")  # Divide by 2 for one-tailed test
print(f"  Result: {'Proposition models show higher confidence' if p_value/2 < 0.05 and t_stat > 0 else 'No significant difference in confidence'}")

# Mann-Whitney U test (non-parametric alternative)
u_stat, mw_p_value = mannwhitneyu(prop_opening_confidences, opp_opening_confidences, alternative='two-sided')
print("\nMann-Whitney U test (non-parametric alternative):")
print(f"  U-statistic: {u_stat}")
print(f"  p-value: {mw_p_value:.4f}")
print(f"  Result: {'Significant difference in confidence distributions' if mw_p_value < 0.05 else 'No significant difference in confidence distributions'}")

# Test 3: ANCOVA-like analysis using linear regression
# Combine data for regression
sides = [0] * len(prop_opening_confidences) + [1] * len(opp_opening_confidences)  # 0 for proposition, 1 for opposition
confidences = prop_opening_confidences + opp_opening_confidences
outcomes = prop_win_outcomes + opp_win_outcomes

# Add constant term for intercept
X = sm.add_constant(np.column_stack((sides, confidences)))
model = sm.OLS(outcomes, X)
results = model.fit()

print("\nRegression analysis (ANCOVA-like approach):")
print("  Model: Win ~ Intercept + Side + Confidence")
print(f"  R-squared: {results.rsquared:.4f}")
print("  Coefficients:")
print(f"    Intercept: {results.params[0]:.4f} (p={results.pvalues[0]:.4f})")
print(f"    Side (Opp=1): {results.params[1]:.4f} (p={results.pvalues[1]:.4f})")
print(f"    Confidence: {results.params[2]:.4f} (p={results.pvalues[2]:.4f})")

side_effect = "has" if results.pvalues[1] < 0.05 else "does not have"
confidence_effect = "is" if results.pvalues[2] < 0.05 else "is not"
print(f"  Interpretation: After controlling for confidence, debate side {side_effect} a significant effect on winning.")
print(f"                  Confidence {confidence_effect} a significant predictor of winning.")

if results.pvalues[1] < 0.05 and results.params[1] > 0:
    print("  This suggests Opposition has an advantage even after accounting for confidence differences.")
elif results.pvalues[1] < 0.05 and results.params[1] < 0:
    print("  This suggests Proposition has an advantage after accounting for confidence differences.")

# Model-specific overconfidence analysis
print("\n--- Model-Specific Overconfidence Analysis ---")
model_data = {}

for debate_data in all_debates:
    debate = debate_data["debate"]
    if not debate.debator_bets or not debate.judge_results:
        continue

    # Determine winner
    winner_counts = {"proposition": 0, "opposition": 0}
    for result in debate.judge_results:
        winner_counts[result.winner] += 1

    winner = "proposition" if winner_counts["proposition"] > winner_counts["opposition"] else "opposition"

    # Process proposition
    prop_model = debate.proposition_model
    if prop_model not in model_data:
        model_data[prop_model] = {"confidences": [], "outcomes": []}

    # Process opposition
    opp_model = debate.opposition_model
    if opp_model not in model_data:
        model_data[opp_model] = {"confidences": [], "outcomes": []}

    # Add confidences and outcomes for both sides
    for bet in debate.debator_bets:
        if bet.speech_type == SpeechType.OPENING:
            if bet.side == Side.PROPOSITION:
                model_data[prop_model]["confidences"].append(bet.amount)
                model_data[prop_model]["outcomes"].append(1 if winner == "proposition" else 0)
            elif bet.side == Side.OPPOSITION:
                model_data[opp_model]["confidences"].append(bet.amount)
                model_data[opp_model]["outcomes"].append(1 if winner == "opposition" else 0)

# Run t-test for each model
print("\nOverconfidence by model (t-test comparing confidence to win rate):")
for model, data in model_data.items():
    if len(data["confidences"]) >= 5:  # Only test if we have enough samples
        confs = data["confidences"]
        outcomes = [x*100 for x in data["outcomes"]]  # Scale to percentage
        t_stat, p_value = ttest_rel(confs, outcomes)
        mean_diff = np.mean(np.array(confs) - np.array(outcomes))
        print(f"{model}:")
        print(f"  Samples: {len(confs)}")
        print(f"  Mean confidence: {np.mean(confs):.2f}%")
        print(f"  Win rate: {np.mean(data['outcomes'])*100:.2f}%")
        print(f"  Mean difference: {mean_diff:.2f}%")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value/2:.4f} (one-tailed)")
        if p_value/2 < 0.05:
            if mean_diff > 0:
                print("  Result: OVERCONFIDENT (statistically significant)")
            else:
                print("  Result: UNDERCONFIDENT (statistically significant)")
        else:
            print("  Result: Well-calibrated (no significant difference)")
        print("")
