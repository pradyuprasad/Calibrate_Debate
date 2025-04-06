#!/usr/bin/env python3
"""
debate_analysis.py - Analyzes debate results with statistical testing and error bars
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.stats import fisher_exact, ttest_1samp, mannwhitneyu

from core.models import DebateTotal, Side, SpeechType

def wilson_score_interval(wins, n, z=1.96):
    """Calculate Wilson score interval for binomial proportion."""
    if n == 0:
        return 0, 0, 0

    p = wins / n
    denominator = 1 + z**2/n
    center = (p + z**2/(2*n))/denominator
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))/denominator

    return p, max(0, center - spread), min(1, center + spread)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("debate_analysis")

# Set the tournament directories
tournament_dirs = [
    Path("tournament/bet_tournament_20250316_1548"),
    Path("tournament/bet_tournament_20250317_1059")
]

# Find all debate files
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
            except Exception as e:
                logger.error(f"Failed to load debate {debate_path}: {str(e)}")

logger.info(f"Loaded {len(debates)} debates across {len(tournament_dirs)} tournaments")

# Collect calibration data
calibration_data = []

for debate_data in debates:
    debate = debate_data["debate"]
    if not debate.debator_bets or not debate.judge_results:
        continue

    # Determine winner
    prop_votes = sum(1 for result in debate.judge_results if result.winner == "proposition")
    opp_votes = sum(1 for result in debate.judge_results if result.winner == "opposition")

    if prop_votes == 0 and opp_votes == 0:
        continue

    winner = "proposition" if prop_votes > opp_votes else "opposition"

    # Get final confidence bets
    prop_confidence = None
    opp_confidence = None

    # Try closing confidence first
    for bet in debate.debator_bets:
        if bet.speech_type == SpeechType.CLOSING:
            if bet.side == Side.PROPOSITION:
                prop_confidence = bet.amount
            elif bet.side == Side.OPPOSITION:
                opp_confidence = bet.amount

    # Fall back to opening confidence
    if prop_confidence is None or opp_confidence is None:
        for bet in debate.debator_bets:
            if bet.speech_type == SpeechType.OPENING:
                if bet.side == Side.PROPOSITION and prop_confidence is None:
                    prop_confidence = bet.amount
                elif bet.side == Side.OPPOSITION and opp_confidence is None:
                    opp_confidence = bet.amount

    if prop_confidence is not None and opp_confidence is not None:
        calibration_data.extend([
            {
                "model": debate.proposition_model,
                "role": "proposition",
                "confidence": prop_confidence,
                "won": winner == "proposition",
                "topic": debate.motion.topic_description
            },
            {
                "model": debate.opposition_model,
                "role": "opposition",
                "confidence": opp_confidence,
                "won": winner == "opposition",
                "topic": debate.motion.topic_description
            }
        ])

# Create confidence bins
confidence_bins = [(50, 60), (61, 70), (71, 80), (81, 90), (91, 100)]
bin_results = defaultdict(lambda: {"count": 0, "wins": 0})

# Populate bins
for entry in calibration_data:
    confidence = entry["confidence"]
    for low, high in confidence_bins:
        if low <= confidence <= high:
            bin_key = f"{low}-{high}%"
            bin_results[bin_key]["count"] += 1
            if entry["won"]:
                bin_results[bin_key]["wins"] += 1

# Calculate calibration metrics
calibration_points = []
for bin_key, results in bin_results.items():
    if results["count"] > 0:
        win_rate, ci_low, ci_high = wilson_score_interval(results["wins"], results["count"])
        bin_center = (int(bin_key.split('-')[0]) + int(bin_key.split('-')[1].rstrip('%'))) / 2

        calibration_points.append({
            "bin": bin_key,
            "confidence": bin_center,
            "win_rate": win_rate * 100,
            "ci_low": ci_low * 100,
            "ci_high": ci_high * 100,
            "count": results["count"],
            "wins": results["wins"]
        })

# Separate proposition and opposition data
prop_data = [entry for entry in calibration_data if entry["role"] == "proposition"]
opp_data = [entry for entry in calibration_data if entry["role"] == "opposition"]

# Calculate role-specific metrics
prop_wins = sum(1 for entry in prop_data if entry["won"])
prop_losses = len(prop_data) - prop_wins
opp_wins = sum(1 for entry in opp_data if entry["won"])
opp_losses = len(opp_data) - opp_wins

prop_win_rate, prop_ci_low, prop_ci_high = wilson_score_interval(prop_wins, len(prop_data))
opp_win_rate, opp_ci_low, opp_ci_high = wilson_score_interval(opp_wins, len(opp_data))

# Statistical tests
# 1. Proposition vs Opposition win rates
contingency_table = [[prop_wins, prop_losses],
                    [opp_wins, opp_losses]]
_, p_value_roles = fisher_exact(contingency_table)

# 2. Calibration errors
errors = np.array([abs(point["confidence"] - point["win_rate"])
                  for point in calibration_points])
t_stat, p_value_calibration = ttest_1samp(errors, 0)

# 3. Proposition vs Opposition calibration
prop_errors = np.array([abs(entry["confidence"] - (100 if entry["won"] else 0))
                       for entry in prop_data])
opp_errors = np.array([abs(entry["confidence"] - (100 if entry["won"] else 0))
                      for entry in opp_data])
stat, p_value_cal_diff = mannwhitneyu(prop_errors, opp_errors, alternative='greater')

# Print results
print("\n=== DEBATE ANALYSIS RESULTS ===")
print("\nOverall Statistics:")
print(f"Total debates analyzed: {len(debates)}")
print(f"Total predictions: {len(calibration_data)}")

print("\nRole Performance:")
print(f"Proposition: {prop_wins}/{len(prop_data)} wins ({prop_win_rate*100:.1f}% ± {(prop_ci_high-prop_ci_low)*50:.1f}%)")
print(f"Opposition: {opp_wins}/{len(opp_data)} wins ({opp_win_rate*100:.1f}% ± {(opp_ci_high-opp_ci_low)*50:.1f}%)")

print("\nStatistical Tests:")
print(f"Proposition vs Opposition win rate difference: p = {p_value_roles:.4f}")
print(f"Calibration error significance: p = {p_value_calibration:.4f}")
print(f"Proposition vs Opposition calibration difference: p = {p_value_cal_diff:.4f}")

# Create calibration plot
plt.figure(figsize=(12, 8))

# Perfect calibration line
plt.plot([0, 100], [0, 100], 'k--', label='Perfect calibration')

# Plot calibration points with error bars
for point in calibration_points:
    yerr = [[point["win_rate"] - point["ci_low"]],
            [point["ci_high"] - point["win_rate"]]]

    plt.errorbar(point["confidence"], point["win_rate"],
                yerr=yerr,
                fmt='o',
                capsize=5,
                capthick=1,
                elinewidth=1,
                markersize=np.sqrt(point["count"]/2),
                label=f"{point['bin']} (n={point['count']})")

plt.xlabel('Model Confidence (%)')
plt.ylabel('Actual Win Rate (%)')
plt.title('Model Calibration Curve with 95% Confidence Intervals')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Add statistical annotation
plt.figtext(0.02, 0.02,
            f"Role difference p={p_value_roles:.4f}\n"
            f"Calibration error p={p_value_calibration:.4f}",
            bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('debate_calibration_analysis.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'debate_calibration_analysis.png'")
