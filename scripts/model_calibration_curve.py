#!/usr/bin/env python3
"""
model_calibration_curve.py - Analyzes model confidence calibration in debates
"""

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from core.models import DebateTotal, Side, SpeechType

# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_calibration")

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

# Collect model confidence and outcome data
calibration_data = []

for debate_data in debates:
   debate = debate_data["debate"]
   if not debate.debator_bets or not debate.judge_results:
       continue

   # Determine winner
   prop_votes = 0
   opp_votes = 0

   for result in debate.judge_results:
       if result.winner == "proposition":
           prop_votes += 1
       elif result.winner == "opposition":
           opp_votes += 1

   if prop_votes == 0 and opp_votes == 0:
       continue

   winner = "proposition" if prop_votes > opp_votes else "opposition"

   # Get final confidence bets
   prop_confidence = None
   opp_confidence = None

   # Try to get closing confidence first
   for bet in debate.debator_bets:
       if bet.speech_type == SpeechType.CLOSING:
           if bet.side == Side.PROPOSITION:
               prop_confidence = bet.amount
           elif bet.side == Side.OPPOSITION:
               opp_confidence = bet.amount

   # If no closing bets, try opening confidence
   if prop_confidence is None or opp_confidence is None:
       for bet in debate.debator_bets:
           if bet.speech_type == SpeechType.OPENING:
               if bet.side == Side.PROPOSITION and prop_confidence is None:
                   prop_confidence = bet.amount
               elif bet.side == Side.OPPOSITION and opp_confidence is None:
                   opp_confidence = bet.amount

   # Only include debates with complete confidence data
   if prop_confidence is not None and opp_confidence is not None:
       # Record proposition model data
       calibration_data.append({
           "model": debate.proposition_model,
           "role": "proposition",
           "confidence": prop_confidence,
           "won": winner == "proposition",
           "topic": debate.motion.topic_description
       })

       # Record opposition model data
       calibration_data.append({
           "model": debate.opposition_model,
           "role": "opposition",
           "confidence": opp_confidence,
           "won": winner == "opposition",
           "topic": debate.motion.topic_description
       })

# Create calibration curve data
confidence_bins = [(50, 60), (61, 70), (71, 80), (81, 90), (91, 100)]
bin_results = {f"{low}-{high}%": {"count": 0, "wins": 0} for low, high in confidence_bins}

# Additional bins for more granular analysis
detailed_bins = [(50, 55), (56, 60), (61, 65), (66, 70), (71, 75),
               (76, 80), (81, 85), (86, 90), (91, 95), (96, 100)]
detailed_results = {f"{low}-{high}%": {"count": 0, "wins": 0} for low, high in detailed_bins}

# Populate bin data
for entry in calibration_data:
   confidence = entry["confidence"]

   # Standard bins
   for low, high in confidence_bins:
       if low <= confidence <= high:
           bin_key = f"{low}-{high}%"
           bin_results[bin_key]["count"] += 1
           if entry["won"]:
               bin_results[bin_key]["wins"] += 1

   # Detailed bins
   for low, high in detailed_bins:
       if low <= confidence <= high:
           bin_key = f"{low}-{high}%"
           detailed_results[bin_key]["count"] += 1
           if entry["won"]:
               detailed_results[bin_key]["wins"] += 1

# Calculate win rates
calibration_data_points = []
for bin_key, results in bin_results.items():
   if results["count"] > 0:
       win_rate = (results["wins"] / results["count"]) * 100
       bin_center = (int(bin_key.split('-')[0]) + int(bin_key.split('-')[1].rstrip('%'))) / 2
       calibration_data_points.append({
           "bin": bin_key,
           "confidence": bin_center,
           "win_rate": win_rate,
           "count": results["count"],
           "wins": results["wins"]
       })

detailed_data_points = []
for bin_key, results in detailed_results.items():
   if results["count"] > 0:
       win_rate = (results["wins"] / results["count"]) * 100
       bin_center = (int(bin_key.split('-')[0]) + int(bin_key.split('-')[1].rstrip('%'))) / 2
       detailed_data_points.append({
           "bin": bin_key,
           "confidence": bin_center,
           "win_rate": win_rate,
           "count": results["count"],
           "wins": results["wins"]
       })

# Calculate overall calibration error (gap between confidence and actual win rate)
calibration_errors = []
weighted_errors = []

for point in calibration_data_points:
   error = abs(point["confidence"] - point["win_rate"])
   calibration_errors.append(error)
   weighted_errors.append(error * point["count"])

if calibration_errors:
   avg_calibration_error = sum(calibration_errors) / len(calibration_errors)
   weighted_avg_error = sum(weighted_errors) / sum(point["count"] for point in calibration_data_points)
else:
   avg_calibration_error = 0
   weighted_avg_error = 0

# Role-specific analysis
prop_data = [entry for entry in calibration_data if entry["role"] == "proposition"]
opp_data = [entry for entry in calibration_data if entry["role"] == "opposition"]

prop_win_rate = sum(1 for entry in prop_data if entry["won"]) / len(prop_data) * 100 if prop_data else 0
opp_win_rate = sum(1 for entry in opp_data if entry["won"]) / len(opp_data) * 100 if opp_data else 0

prop_avg_conf = sum(entry["confidence"] for entry in prop_data) / len(prop_data) if prop_data else 0
opp_avg_conf = sum(entry["confidence"] for entry in opp_data) / len(opp_data) if opp_data else 0

# Calculate calibration error by role
prop_error = abs(prop_avg_conf - prop_win_rate)
opp_error = abs(opp_avg_conf - opp_win_rate)

# Print results
print("\n=== MODEL CONFIDENCE CALIBRATION ANALYSIS ===")
print(f"Total model predictions analyzed: {len(calibration_data)}")
print(f"Average calibration error: {avg_calibration_error:.2f}%")
print(f"Weighted average calibration error: {weighted_avg_error:.2f}%")

print("\n=== CALIBRATION BY CONFIDENCE BIN ===")
print("Bin | Predictions | Wins | Win Rate | Expected Win Rate | Calibration Error")
print("-" * 80)
for point in sorted(calibration_data_points, key=lambda x: x["confidence"]):
   error = abs(point["confidence"] - point["win_rate"])
   print(f"{point['bin']:7} | {point['count']:11} | {point['wins']:4} | {point['win_rate']:8.2f}% | {point['confidence']:15.2f}% | {error:16.2f}%")

print("\n=== DETAILED CALIBRATION BY CONFIDENCE BIN ===")
print("Bin | Predictions | Wins | Win Rate | Expected Win Rate | Calibration Error")
print("-" * 80)
for point in sorted(detailed_data_points, key=lambda x: x["confidence"]):
   error = abs(point["confidence"] - point["win_rate"])
   print(f"{point['bin']:7} | {point['count']:11} | {point['wins']:4} | {point['win_rate']:8.2f}% | {point['confidence']:15.2f}% | {error:16.2f}%")

print("\n=== CALIBRATION BY ROLE ===")
print(f"Proposition models: {len(prop_data)} predictions, Win rate: {prop_win_rate:.2f}%, Avg confidence: {prop_avg_conf:.2f}%, Error: {prop_error:.2f}%")
print(f"Opposition models: {len(opp_data)} predictions, Win rate: {opp_win_rate:.2f}%, Avg confidence: {opp_avg_conf:.2f}%, Error: {opp_error:.2f}%")

# Model-specific analysis
model_stats = defaultdict(lambda: {"count": 0, "wins": 0, "confidence_sum": 0})
for entry in calibration_data:
   model_name = entry["model"].split('/')[-1]  # Extract just the model name
   model_stats[model_name]["count"] += 1
   if entry["won"]:
       model_stats[model_name]["wins"] += 1
   model_stats[model_name]["confidence_sum"] += entry["confidence"]

print("\n=== MODEL-SPECIFIC CALIBRATION ===")
print("Model | Predictions | Win Rate | Avg Confidence | Calibration Error")
print("-" * 80)
for model, stats in sorted(model_stats.items(), key=lambda x: abs(
   (x[1]["wins"]/x[1]["count"]*100 if x[1]["count"] > 0 else 0) -
   (x[1]["confidence_sum"]/x[1]["count"] if x[1]["count"] > 0 else 0)
), reverse=True):
   count = stats["count"]
   win_rate = (stats["wins"] / count) * 100 if count > 0 else 0
   avg_conf = stats["confidence_sum"] / count if count > 0 else 0
   error = abs(win_rate - avg_conf)
   print(f"{model:20} | {count:11} | {win_rate:8.2f}% | {avg_conf:14.2f}% | {error:16.2f}%")

print("\nAnalysis complete!")

# Create and save calibration curve plot
try:
   plt.figure(figsize=(10, 8))

   # Plot perfect calibration line
   plt.plot([0, 100], [0, 100], 'k--', label='Perfect calibration')

   # Plot actual calibration points
   conf_values = [point["confidence"] for point in calibration_data_points]
   win_rates = [point["win_rate"] for point in calibration_data_points]
   sizes = [point["count"] * 10 for point in calibration_data_points]  # Scale point size by count

   plt.scatter(conf_values, win_rates, s=sizes, alpha=0.7, label='Observed calibration')

   # Add labels for each point
   for point in calibration_data_points:
       plt.annotate(f"{point['bin']}\n({point['count']} predictions)",
                   (point["confidence"], point["win_rate"]),
                   textcoords="offset points",
                   xytext=(0,10),
                   ha='center')

   # Add additional lines and annotations
   plt.axhline(y=50, color='r', linestyle='-', alpha=0.3, label='Random chance (50%)')

   # Set labels and title
   plt.xlabel('Model Confidence (%)')
   plt.ylabel('Actual Win Rate (%)')
   plt.title('Model Calibration Curve: Confidence vs. Actual Win Rate')

   # Set axis limits and grid
   plt.xlim(45, 105)
   plt.ylim(-5, 105)
   plt.grid(True, alpha=0.3)

   # Add legend
   plt.legend(loc='upper left')

   # Add annotation about calibration error
   plt.figtext(0.5, 0.01,
               f"Average calibration error: {avg_calibration_error:.2f}% | Weighted error: {weighted_avg_error:.2f}%",
               ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

   # Save plot
   plt.tight_layout()
   plt.savefig('model_calibration_curve.png', dpi=300)
   print("Calibration curve plot saved to 'model_calibration_curve.png'")
except Exception as e:
   print(f"Could not create plot: {e}")
