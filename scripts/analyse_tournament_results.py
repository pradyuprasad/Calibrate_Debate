#!/usr/bin/env python3
"""
Script to extract detailed quantitative metrics from debate tournament data.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict

from core.models import DebateTotal, Side, SpeechType
from scripts.utils import sanitize_model_name

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("quantitative_analysis")

# Set the tournament directory
tournament_dir = Path("tournament/bet_tournament_20250316_1548")

# Load tournament results
tournament_results_path = tournament_dir / "tournament_results.json"
with open(tournament_results_path, "r") as f:
    tournament_results = json.load(f)
    model_stats = tournament_results.get("model_stats", {})

# Print all available models with indices
print("\n=== AVAILABLE MODELS ===")
all_models = list(model_stats.keys())
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
original_model_count = len(model_stats)
for excluded_model in EXCLUDED_MODELS:
    if excluded_model in model_stats:
        logger.info(f"Excluding model from stats: {excluded_model}")
        del model_stats[excluded_model]

if EXCLUDED_MODELS:
    logger.info(f"Excluded {original_model_count - len(model_stats)} models from analysis")

# Find all debate files
debates = []
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

            debates.append({
                "round": round_num,
                "path": debate_path,
                "debate": debate,
                "prop_model": debate.proposition_model,
                "opp_model": debate.opposition_model,
                "topic": debate.motion.topic_description
            })
            logger.info(f"Loaded debate: {debate_path.name}")
        except Exception as e:
            logger.error(f"Failed to load debate {debate_path}: {str(e)}")

logger.info(f"Loaded {len(debates)} debates across {len(round_dirs)} rounds")

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

# Extract quantitative data from debates
for debate_data in debates:
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
print("\n=== JUDGE AGREEMENT ===")
total_judged = judge_agreement_count["unanimous"] + judge_agreement_count["split"]
if total_judged > 0:
    unanimous_pct = (judge_agreement_count["unanimous"] / total_judged) * 100
    split_pct = (judge_agreement_count["split"] / total_judged) * 100
    print(f"Unanimous decisions: {judge_agreement_count['unanimous']}/{total_judged} ({unanimous_pct:.1f}%)")
    print(f"Split decisions: {judge_agreement_count['split']}/{total_judged} ({split_pct:.1f}%)")

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

print("\nQuantitative analysis complete!")


# Add this section after the other analyses

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
for debate_data in debates:
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

