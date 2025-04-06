#!/usr/bin/env python3
"""
Simple script to analyze bet tournament results by loading debate JSON files.
Prints all analysis to command line, with option to exclude specific models by index.
Supports multiple tournament directories.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict

from core.models import DebateTotal, Side, SpeechType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("analysis")

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
            current_model_stats = tournament_results.get("model_stats", {})

            # For existing models, merge their stats
            for model, stats in current_model_stats.items():
                if model in combined_model_stats:
                    # Combine wins, losses, rounds played
                    combined_model_stats[model]["wins"] += stats["wins"]
                    combined_model_stats[model]["losses"] += stats["losses"]
                    combined_model_stats[model]["rounds_played"] += stats["rounds_played"]
                    combined_model_stats[model]["total_margin"] += stats["total_margin"]

                    # Combine bet histories
                    if "bet_history" in stats:
                        if "bet_history" not in combined_model_stats[model]:
                            combined_model_stats[model]["bet_history"] = []
                        combined_model_stats[model]["bet_history"].extend(stats["bet_history"])
                else:
                    # New model, just add it
                    combined_model_stats[model] = stats.copy()

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
        del combined_model_stats[excluded_model]

if EXCLUDED_MODELS:
    logger.info(f"Excluded {original_model_count - len(combined_model_stats)} models from analysis")

# Find all debate files across all tournament directories
debates = []
total_rounds = 0

for tournament_dir in tournament_dirs:
    round_dirs = [d for d in tournament_dir.glob("round_*") if d.is_dir()]
    total_rounds += len(round_dirs)

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
                    "topic": debate.motion.topic_description,
                    "tournament": tournament_dir.name
                })
                logger.info(f"Loaded debate: {debate_path.name} from {tournament_dir.name}")
            except Exception as e:
                logger.error(f"Failed to load debate {debate_path}: {str(e)}")

# Analyze model performance
logger.info(f"Loaded {len(debates)} debates across {total_rounds} rounds from {len(tournament_dirs)} tournaments")

# 1. Overall model rankings
print("\n=== OVERALL MODEL RANKINGS ===")
for model, stats in sorted(
    combined_model_stats.items(),
    key=lambda x: (x[1]["wins"] - x[1]["losses"], x[1]["total_margin"]),
    reverse=True
):
    win_rate = (stats["wins"] / stats["rounds_played"] * 100) if stats["rounds_played"] > 0 else 0
    print(f"{model}: {stats['wins']}W-{stats['losses']}L (Win rate: {win_rate:.1f}%, Margin: {stats['total_margin']:.2f})")

# 2. Analyze bet patterns
print("\n=== BETTING PATTERNS ===")
for model, stats in combined_model_stats.items():
    bet_histories = stats.get("bet_history", [])
    if not bet_histories:
        continue

    increased = 0
    decreased = 0
    unchanged = 0
    total_debates = len(bet_histories)

    # ASCII chart for confidence evolution
    print(f"\n{model} confidence patterns:")
    for i, bets in enumerate(bet_histories):
        if not bets:
            continue

        # Print a simple ASCII chart of confidence evolution
        bet_str = " → ".join([str(bet) for bet in bets])
        change = ""
        if len(bets) >= 2:
            if bets[-1] > bets[0]:
                increased += 1
                change = f"↑ +{bets[-1] - bets[0]}"
            elif bets[-1] < bets[0]:
                decreased += 1
                change = f"↓ {bets[-1] - bets[0]}"
            else:
                unchanged += 1
                change = "↔ 0"

        print(f"  Debate {i+1}: {bet_str} {change}")

    if total_debates > 0:
        print(f"  Summary: {increased}/{total_debates} increased ({increased/total_debates*100:.1f}%), "
              f"{decreased}/{total_debates} decreased ({decreased/total_debates*100:.1f}%), "
              f"{unchanged}/{total_debates} unchanged ({unchanged/total_debates*100:.1f}%)")

# 3. Topic analysis - which topics led to biggest confidence changes
topic_confidence_changes = defaultdict(list)

for debate_data in debates:
    debate = debate_data["debate"]
    if not debate.debator_bets:
        continue

    # Track initial and final bets for prop and opp
    prop_bets = {"opening": None, "closing": None}
    opp_bets = {"opening": None, "closing": None}

    for bet in debate.debator_bets:
        if bet.side == Side.PROPOSITION:
            if bet.speech_type == SpeechType.OPENING:
                prop_bets["opening"] = bet.amount
            elif bet.speech_type == SpeechType.CLOSING:
                prop_bets["closing"] = bet.amount
        elif bet.side == Side.OPPOSITION:
            if bet.speech_type == SpeechType.OPENING:
                opp_bets["opening"] = bet.amount
            elif bet.speech_type == SpeechType.CLOSING:
                opp_bets["closing"] = bet.amount

    # Calculate changes if we have both opening and closing
    topic = debate.motion.topic_description
    if prop_bets["opening"] is not None and prop_bets["closing"] is not None:
        change = prop_bets["closing"] - prop_bets["opening"]
        topic_confidence_changes[topic].append((change, debate_data["tournament"]))

    if opp_bets["opening"] is not None and opp_bets["closing"] is not None:
        change = opp_bets["closing"] - opp_bets["opening"]
        topic_confidence_changes[topic].append((change, debate_data["tournament"]))

# Print topics with biggest average confidence changes
if topic_confidence_changes:
    print("\n=== TOPICS WITH BIGGEST CONFIDENCE CHANGES ===")

    # Process the topic changes with tournament info
    topic_stats = {}
    for topic, changes_with_tourney in topic_confidence_changes.items():
        changes = [change for change, _ in changes_with_tourney]
        avg_change = sum(changes) / len(changes) if changes else 0
        tournaments = set(tourney for _, tourney in changes_with_tourney)
        topic_stats[topic] = {
            "avg_change": avg_change,
            "abs_avg_change": abs(avg_change),
            "tournaments": ", ".join(t.split('/')[-1] for t in tournaments)
        }

    sorted_topics = sorted(topic_stats.items(), key=lambda x: x[1]["abs_avg_change"], reverse=True)

    for topic, stats in sorted_topics:
        avg_change = stats["avg_change"]
        direction = "↑" if avg_change > 0 else "↓" if avg_change < 0 else "↔"
        tournaments = stats["tournaments"]
        print(f"Topic: {topic} - Avg change: {avg_change:.2f} {direction} (Found in: {tournaments})")

# 4. Analyze correlation between initial confidence and winning
print("\n=== CORRELATION: INITIAL CONFIDENCE VS WINNING ===")
high_initial_confidence_wins = 0
high_initial_confidence_total = 0
low_initial_confidence_wins = 0
low_initial_confidence_total = 0

# Add counters for both sides high confidence
both_over_50_count = 0
both_over_75_count = 0
total_debates_with_bets = 0

# Add counters for side-specific stats
prop_wins = 0
opp_wins = 0
total_debates_with_winners = 0

for debate_data in debates:
    debate = debate_data["debate"]
    if not debate.debator_bets or not debate.judge_results:
        continue

    # Determine winner from judge results
    winner_counts = {"proposition": 0, "opposition": 0}
    for result in debate.judge_results:
        winner_counts[result.winner] += 1

    winner = "proposition" if winner_counts["proposition"] > winner_counts["opposition"] else "opposition"

    # Track overall win rates by side
    total_debates_with_winners += 1
    if winner == "proposition":
        prop_wins += 1
    else:
        opp_wins += 1

    # Get initial confidence bets
    prop_initial = None
    opp_initial = None

    for bet in debate.debator_bets:
        if bet.speech_type == SpeechType.OPENING:
            if bet.side == Side.PROPOSITION:
                prop_initial = bet.amount
            elif bet.side == Side.OPPOSITION:
                opp_initial = bet.amount

    if prop_initial is not None and opp_initial is not None:
        total_debates_with_bets += 1

        # Check for both sides having high confidence
        if prop_initial > 50 and opp_initial > 50:
            both_over_50_count += 1

        if prop_initial > 75 and opp_initial > 75:
            both_over_75_count += 1

        # Check proposition
        if prop_initial > 50:
            high_initial_confidence_total += 1
            if winner == "proposition":
                high_initial_confidence_wins += 1
        else:
            low_initial_confidence_total += 1
            if winner == "proposition":
                low_initial_confidence_wins += 1

        # Check opposition
        if opp_initial > 50:
            high_initial_confidence_total += 1
            if winner == "opposition":
                high_initial_confidence_wins += 1
        else:
            low_initial_confidence_total += 1
            if winner == "opposition":
                low_initial_confidence_wins += 1

# Print correlation results
if high_initial_confidence_total > 0 and low_initial_confidence_total > 0:
    high_win_rate = (high_initial_confidence_wins / high_initial_confidence_total) * 100
    low_win_rate = (low_initial_confidence_wins / low_initial_confidence_total) * 100

    print(f"High initial confidence (>50): {high_initial_confidence_wins}/{high_initial_confidence_total} wins ({high_win_rate:.1f}%)")
    print(f"Low initial confidence (≤50): {low_initial_confidence_wins}/{low_initial_confidence_total} wins ({low_win_rate:.1f}%)")
    print(f"Difference: {abs(high_win_rate - low_win_rate):.1f}%")

# Print statistics about both sides having high confidence
print("\n=== MUTUAL HIGH CONFIDENCE STATISTICS ===")
if total_debates_with_bets > 0:
    both_over_50_pct = (both_over_50_count / total_debates_with_bets) * 100
    both_over_75_pct = (both_over_75_count / total_debates_with_bets) * 100

    print(f"Debates where both sides bet >50: {both_over_50_count}/{total_debates_with_bets} ({both_over_50_pct:.1f}%)")
    print(f"Debates where both sides bet >75: {both_over_75_count}/{total_debates_with_bets} ({both_over_75_pct:.1f}%)")

# Print side win statistics
print("\n=== SIDE WIN STATISTICS ===")
if total_debates_with_winners > 0:
    prop_win_rate = (prop_wins / total_debates_with_winners) * 100
    opp_win_rate = (opp_wins / total_debates_with_winners) * 100

    print(f"Proposition wins: {prop_wins}/{total_debates_with_winners} ({prop_win_rate:.1f}%)")
    print(f"Opposition wins: {opp_wins}/{total_debates_with_winners} ({opp_win_rate:.1f}%)")
    print(f"Opposition/Proposition win ratio: {opp_wins/prop_wins:.2f} to 1" if prop_wins > 0 else "Opposition wins all debates")

print("\nAnalysis complete!")
