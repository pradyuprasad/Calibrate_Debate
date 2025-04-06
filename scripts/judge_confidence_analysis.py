#!/usr/bin/env python3
"""
judge_confidence_analysis.py - Analyzes judge confidence patterns in relation to debate outcomes
"""

import logging
import numpy as np
from pathlib import Path

from core.models import DebateTotal, Side, SpeechType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("judge_confidence_analysis")

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

# Analyze judge confidence in relation to debate outcomes
judge_confidence_data = []

for debate_data in debates:
    debate = debate_data["debate"]
    if not debate.judge_results:
        continue

    # Get judges votes and confidence
    prop_votes = 0
    opp_votes = 0
    prop_confidence_sum = 0
    opp_confidence_sum = 0
    total_confidence = 0

    for result in debate.judge_results:
        if hasattr(result, 'confidence') and result.confidence is not None:
            total_confidence += result.confidence
            if result.winner == "proposition":
                prop_votes += 1
                prop_confidence_sum += result.confidence
            elif result.winner == "opposition":
                opp_votes += 1
                opp_confidence_sum += result.confidence

    # Determine overall winner
    winner = "proposition" if prop_votes > opp_votes else "opposition"
    loser = "opposition" if winner == "proposition" else "proposition"

    # Calculate contentiousness measures
    if total_confidence > 0:
        # Normalized confidence in loser (higher = more contentious)
        loser_confidence_sum = opp_confidence_sum if winner == "proposition" else prop_confidence_sum
        normalized_loser_confidence = loser_confidence_sum / total_confidence

        # Calculate vote split (0 = unanimous, 0.5 = maximally split)
        vote_split = min(prop_votes, opp_votes) / (prop_votes + opp_votes)

        # Combined contentiousness measure (weighs both vote split and confidence)
        contentiousness = (normalized_loser_confidence + vote_split) / 2

        # Get model confidence data
        prop_model_confidence = None
        opp_model_confidence = None

        # Try to get closing confidence first
        for bet in debate.debator_bets:
            if bet.speech_type == SpeechType.CLOSING:
                if bet.side == Side.PROPOSITION:
                    prop_model_confidence = bet.amount
                elif bet.side == Side.OPPOSITION:
                    opp_model_confidence = bet.amount

        # If no closing bets, try opening confidence
        if prop_model_confidence is None or opp_model_confidence is None:
            for bet in debate.debator_bets:
                if bet.speech_type == SpeechType.OPENING:
                    if bet.side == Side.PROPOSITION and prop_model_confidence is None:
                        prop_model_confidence = bet.amount
                    elif bet.side == Side.OPPOSITION and opp_model_confidence is None:
                        opp_model_confidence = bet.amount

        # Only include debates with complete data
        if prop_model_confidence is not None and opp_model_confidence is not None:
            # Winner's model confidence
            winner_model_confidence = prop_model_confidence if winner == "proposition" else opp_model_confidence

            # Loser's model confidence
            loser_model_confidence = opp_model_confidence if winner == "proposition" else prop_model_confidence

            # Store debate data
            judge_confidence_data.append({
                "topic": debate.motion.topic_description,
                "winner": winner,
                "winner_model": debate.proposition_model if winner == "proposition" else debate.opposition_model,
                "loser_model": debate.opposition_model if winner == "proposition" else debate.proposition_model,
                "prop_votes": prop_votes,
                "opp_votes": opp_votes,
                "vote_split": vote_split,
                "prop_confidence_sum": prop_confidence_sum,
                "opp_confidence_sum": opp_confidence_sum,
                "normalized_loser_confidence": normalized_loser_confidence,
                "contentiousness": contentiousness,
                "winner_model_confidence": winner_model_confidence,
                "loser_model_confidence": loser_model_confidence,
                "judge_confidence_diff": (prop_confidence_sum if winner == "proposition" else opp_confidence_sum) -
                                        (opp_confidence_sum if winner == "proposition" else prop_confidence_sum)
            })

# Print results
print("\n=== JUDGE CONFIDENCE ANALYSIS ===")
print(f"Total debates with judge confidence data: {len(judge_confidence_data)}")

# Calculate correlations
if len(judge_confidence_data) >= 3:
    contentiousness_values = [data["contentiousness"] for data in judge_confidence_data]
    winner_conf_values = [data["winner_model_confidence"] for data in judge_confidence_data]
    loser_conf_values = [data["loser_model_confidence"] for data in judge_confidence_data]
    judge_conf_diff_values = [data["judge_confidence_diff"] for data in judge_confidence_data]

    cont_winner_corr = np.corrcoef(contentiousness_values, winner_conf_values)[0, 1]
    cont_loser_corr = np.corrcoef(contentiousness_values, loser_conf_values)[0, 1]
    cont_judge_diff_corr = np.corrcoef(contentiousness_values, judge_conf_diff_values)[0, 1]

    print("\n=== CORRELATIONS ===")
    print(f"Correlation between contentiousness and winner model confidence: {cont_winner_corr:.3f}")
    print(f"Correlation between contentiousness and loser model confidence: {cont_loser_corr:.3f}")
    print(f"Correlation between contentiousness and judge confidence difference: {cont_judge_diff_corr:.3f}")

    # Group debates by contentiousness
    high_contentious = [d for d in judge_confidence_data if d["contentiousness"] >= 0.33]
    med_contentious = [d for d in judge_confidence_data if 0.15 <= d["contentiousness"] < 0.33]
    low_contentious = [d for d in judge_confidence_data if d["contentiousness"] < 0.15]

    # Calculate average judge confidence by contentiousness
    def calc_averages(debates):
        if not debates:
            return {
                "count": 0,
                "avg_winner_model_conf": 0,
                "avg_loser_model_conf": 0,
                "avg_judge_winner_conf": 0,
                "avg_judge_loser_conf": 0,
                "avg_judge_conf_diff": 0
            }

        return {
            "count": len(debates),
            "avg_winner_model_conf": sum(d["winner_model_confidence"] for d in debates) / len(debates),
            "avg_loser_model_conf": sum(d["loser_model_confidence"] for d in debates) / len(debates),
            "avg_judge_winner_conf": sum((d["prop_confidence_sum"] if d["winner"] == "proposition" else d["opp_confidence_sum"]) /
                               (d["prop_votes"] if d["winner"] == "proposition" else d["opp_votes"])
                               if (d["prop_votes"] if d["winner"] == "proposition" else d["opp_votes"]) > 0
                               else 0 for d in debates) / len(debates),
            "avg_judge_loser_conf": sum((d["opp_confidence_sum"] if d["winner"] == "proposition" else d["prop_confidence_sum"]) /
                              (d["opp_votes"] if d["winner"] == "proposition" else d["prop_votes"])
                              if (d["opp_votes"] if d["winner"] == "proposition" else d["prop_votes"]) > 0
                              else 0 for d in debates) / len(debates),
            "avg_judge_conf_diff": sum(d["judge_confidence_diff"] for d in debates) / len(debates)
        }

    high_contentious_stats = calc_averages(high_contentious)
    med_contentious_stats = calc_averages(med_contentious)
    low_contentious_stats = calc_averages(low_contentious)

    print("\n=== CONFIDENCE PATTERNS BY CONTENTIOUSNESS ===")

    for label, stats in [("High", high_contentious_stats), ("Medium", med_contentious_stats), ("Low", low_contentious_stats)]:
        if stats["count"] > 0:
            print(f"\n{label} contentiousness debates ({stats['count']} debates):")
            print(f"Average winner model confidence: {stats['avg_winner_model_conf']:.2f}")
            print(f"Average loser model confidence: {stats['avg_loser_model_conf']:.2f}")
            print(f"Winner higher than loser by: {stats['avg_winner_model_conf'] - stats['avg_loser_model_conf']:+.2f}")
            print(f"Average judge confidence in winner: {stats['avg_judge_winner_conf']:.2f}")
            print(f"Average judge confidence in loser: {stats['avg_judge_loser_conf']:.2f}")
            print(f"Judge confidence difference: {stats['avg_judge_conf_diff']:+.2f}")

    # Individual debate analysis sorted by contentiousness
    print("\n=== INDIVIDUAL DEBATE ANALYSIS (SORTED BY CONTENTIOUSNESS) ===")
    print("Topic | Contentiousness | Winner | Vote Split | Judge Conf in Winner | Judge Conf in Loser | Winner Model Conf | Loser Model Conf")
    print("-" * 140)

    for data in sorted(judge_confidence_data, key=lambda x: x["contentiousness"], reverse=True):
        topic_short = data["topic"][:30] + "..." if len(data["topic"]) > 30 else data["topic"].ljust(33)
        winner_model = data["winner_model"].split('/')[-1]

        # Calculate average judge confidence
        avg_winner_judge_conf = data["prop_confidence_sum"] / data["prop_votes"] if data["winner"] == "proposition" and data["prop_votes"] > 0 else \
                               data["opp_confidence_sum"] / data["opp_votes"] if data["winner"] == "opposition" and data["opp_votes"] > 0 else 0

        avg_loser_judge_conf = data["opp_confidence_sum"] / data["opp_votes"] if data["winner"] == "proposition" and data["opp_votes"] > 0 else \
                              data["prop_confidence_sum"] / data["prop_votes"] if data["winner"] == "opposition" and data["prop_votes"] > 0 else 0

        print(f"{topic_short} | {data['contentiousness']:15.3f} | {data['winner']:6} | {data['vote_split']:10.2f} | " +
              f"{avg_winner_judge_conf:19.2f} | {avg_loser_judge_conf:18.2f} | {data['winner_model_confidence']:16.1f} | " +
              f"{data['loser_model_confidence']:15.1f}")

print("\nAnalysis complete!")
