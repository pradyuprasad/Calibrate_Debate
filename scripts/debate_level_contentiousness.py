#!/usr/bin/env python3
"""
correlation_confidence_win_margin.py - Analyzes correlation between model confidence and win margins
"""

import logging
from pathlib import Path

import numpy as np

from core.models import DebateTotal, Side, SpeechType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("confidence_win_margin_correlation")

# Set the tournament directories
tournament_dirs = [
    Path("tournament/bet_tournament_20250316_1548"),
    Path("tournament/bet_tournament_20250317_1059")
]

# Find all debate files across all tournament directories
debates = []

for tournament_dir in tournament_dirs:
    round_dirs = [d for d in tournament_dir.glob("round_*") if d.is_dir()]

    for round_dir in round_dirs:
        round_num = int(round_dir.name.split("_")[1])
        for debate_path in round_dir.glob("*.json"):
            try:
                debate = DebateTotal.load_from_json(debate_path)
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

logger.info(f"Loaded {len(debates)} debates across {len(tournament_dirs)} tournaments")

# Analyze correlation between confidence and win margin
confidence_margin_data = []

for debate_data in debates:
    debate = debate_data["debate"]
    if not debate.debator_bets or not debate.judge_results:
        continue

    # Calculate judge agreement and win margin
    prop_votes = 0
    opp_votes = 0

    for result in debate.judge_results:
        if result.winner == "proposition":
            prop_votes += 1
        elif result.winner == "opposition":
            opp_votes += 1

    total_votes = prop_votes + opp_votes
    if total_votes == 0:  # Skip if no valid votes
        continue

    # Calculate win margin as percentage difference
    # Positive = proposition win margin, negative = opposition win margin
    win_margin = ((prop_votes - opp_votes) / total_votes) * 100

    # Get final confidence bets
    prop_confidence = None
    opp_confidence = None

    # First try to get closing confidence
    for bet in debate.debator_bets:
        if bet.speech_type == SpeechType.CLOSING:
            if bet.side == Side.PROPOSITION:
                prop_confidence = bet.amount
            elif bet.side == Side.OPPOSITION:
                opp_confidence = bet.amount

    # If no closing bets, try to get opening confidence
    if prop_confidence is None or opp_confidence is None:
        for bet in debate.debator_bets:
            if bet.speech_type == SpeechType.OPENING:
                if bet.side == Side.PROPOSITION and prop_confidence is None:
                    prop_confidence = bet.amount
                elif bet.side == Side.OPPOSITION and opp_confidence is None:
                    opp_confidence = bet.amount

    # Only include debates with complete confidence data
    if prop_confidence is None or opp_confidence is None:
        continue

    # Normalize confidence values to account for which side won
    # For proposition winner, we use prop confidence directly
    # For opposition winner, we use (100 - opp confidence) to represent error
    winner = "proposition" if prop_votes > opp_votes else "opposition"

    if winner == "proposition":
        confidence_of_winner = prop_confidence
        margin = win_margin  # Already positive for proposition
    else:
        confidence_of_winner = opp_confidence
        margin = -win_margin  # Make positive for opposition

    # Store data for correlation analysis
    confidence_margin_data.append({
        "topic": debate.motion.topic_description,
        "winner": winner,
        "margin": abs(margin),  # Always positive, representing magnitude
        "winner_confidence": confidence_of_winner,
        "winner_model": debate.proposition_model if winner == "proposition" else debate.opposition_model,
        "prop_confidence": prop_confidence,
        "opp_confidence": opp_confidence,
        "prop_votes": prop_votes,
        "opp_votes": opp_votes
    })

# Print basic statistics
print("\n=== CONFIDENCE VS WIN MARGIN ANALYSIS ===")
print(f"Total debates with complete data: {len(confidence_margin_data)}")

# Analyze confidence accuracy
correct_confidence = 0
for data in confidence_margin_data:
    if (data["winner"] == "proposition" and data["prop_confidence"] > data["opp_confidence"]) or \
       (data["winner"] == "opposition" and data["opp_confidence"] > data["prop_confidence"]):
        correct_confidence += 1

confidence_accuracy = (correct_confidence / len(confidence_margin_data)) * 100 if confidence_margin_data else 0
print(f"Confidence predicted winner correctly: {correct_confidence}/{len(confidence_margin_data)} ({confidence_accuracy:.1f}%)")

# Calculate correlation
margins = [data["margin"] for data in confidence_margin_data]
winner_confidences = [data["winner_confidence"] for data in confidence_margin_data]

if len(margins) >= 3 and len(winner_confidences) >= 3:
    correlation = np.corrcoef(margins, winner_confidences)[0, 1]
    print(f"\nCorrelation between win margin and winner's confidence: {correlation:.3f}")

    # Group by confidence level
    high_confidence = [data for data in confidence_margin_data if data["winner_confidence"] > 75]
    medium_confidence = [data for data in confidence_margin_data if 50 < data["winner_confidence"] <= 75]
    low_confidence = [data for data in confidence_margin_data if data["winner_confidence"] <= 50]

    # Calculate average margins
    high_avg_margin = sum(data["margin"] for data in high_confidence) / len(high_confidence) if high_confidence else 0
    medium_avg_margin = sum(data["margin"] for data in medium_confidence) / len(medium_confidence) if medium_confidence else 0
    low_avg_margin = sum(data["margin"] for data in low_confidence) / len(low_confidence) if low_confidence else 0

    print("\n=== AVERAGE WIN MARGINS BY CONFIDENCE LEVEL ===")
    print(f"High confidence (>75%): {len(high_confidence)} debates, avg margin: {high_avg_margin:.2f}%")
    print(f"Medium confidence (51-75%): {len(medium_confidence)} debates, avg margin: {medium_avg_margin:.2f}%")
    print(f"Low confidence (≤50%): {len(low_confidence)} debates, avg margin: {low_avg_margin:.2f}%")

    # Analyze unanimous vs split decisions
    unanimous = [data for data in confidence_margin_data if data["margin"] == 100]  # 100% margin = unanimous
    split = [data for data in confidence_margin_data if data["margin"] < 100]

    unanimous_avg_conf = sum(data["winner_confidence"] for data in unanimous) / len(unanimous) if unanimous else 0
    split_avg_conf = sum(data["winner_confidence"] for data in split) / len(split) if split else 0

    print("\n=== CONFIDENCE BY DECISION TYPE ===")
    print(f"Unanimous decisions: {len(unanimous)} debates, avg winner confidence: {unanimous_avg_conf:.2f}%")
    print(f"Split decisions: {len(split)} debates, avg winner confidence: {split_avg_conf:.2f}%")

    # Calculate loser confidence stats
    loser_confidences = []
    for data in confidence_margin_data:
        loser_conf = data["opp_confidence"] if data["winner"] == "proposition" else data["prop_confidence"]
        loser_confidences.append(loser_conf)

    avg_loser_conf = sum(loser_confidences) / len(loser_confidences) if loser_confidences else 0
    print(f"\nAverage loser confidence: {avg_loser_conf:.2f}%")
    print(f"Average winner confidence: {sum(winner_confidences) / len(winner_confidences):.2f}%")
    print(f"Winner-loser confidence difference: {(sum(winner_confidences) / len(winner_confidences)) - avg_loser_conf:.2f}%")

    # List individual debates sorted by margin
    print("\n=== INDIVIDUAL DEBATE ANALYSIS (SORTED BY MARGIN) ===")
    print("Topic | Win Margin | Winner Confidence | Loser Confidence | Winner Model")
    print("-" * 100)

    for data in sorted(confidence_margin_data, key=lambda x: x["margin"], reverse=True):
        topic_short = data["topic"][:30] + "..." if len(data["topic"]) > 30 else data["topic"].ljust(33)
        winner_model = data["winner_model"].split('/')[-1]
        loser_conf = data["opp_confidence"] if data["winner"] == "proposition" else data["prop_confidence"]

        print(f"{topic_short} | {data['margin']:10.1f}% | {data['winner_confidence']:17.1f}% | {loser_conf:16.1f}% | {winner_model}")

else:
    print("\nInsufficient data for correlation analysis")

print("\nAnalysis complete!")
