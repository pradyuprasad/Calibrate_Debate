#!/usr/bin/env python3
"""
Script to analyze whether specific models tend to express higher confidence
regardless of topic difficulty.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np

from core.models import DebateTotal, Side, SpeechType

# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_confidence_analysis")

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

# Initialize data structures for analysis
model_confidence_by_topic = defaultdict(lambda: defaultdict(list))
model_baseline_confidence = defaultdict(list)
model_win_rates = defaultdict(lambda: {"wins": 0, "total": 0})
topic_difficulty = {}

# Calculate topic difficulty based on judge disagreement
for debate_data in debates:
   debate = debate_data["debate"]
   topic = debate.motion.topic_description

   if not debate.judge_results:
       continue

   # Count judge disagreements
   winner_counts = {"proposition": 0, "opposition": 0}
   for result in debate.judge_results:
       winner_counts[result.winner] += 1

   total_judges = len(debate.judge_results)
   max_agreement = max(winner_counts.values())

   # Track topic difficulty
   if topic not in topic_difficulty:
       topic_difficulty[topic] = {"disagreements": 0, "total": 0}

   topic_difficulty[topic]["total"] += 1
   if max_agreement < total_judges:  # Not unanimous
       topic_difficulty[topic]["disagreements"] += 1

# Calculate difficulty percentages
for topic, data in topic_difficulty.items():
   if data["total"] > 0:
       data["difficulty"] = (data["disagreements"] / data["total"]) * 100
   else:
       data["difficulty"] = 0

# Gather model confidence data
for debate_data in debates:
   debate = debate_data["debate"]
   if not debate.debator_bets or not debate.judge_results:
       continue

   topic = debate.motion.topic_description
   topic_diff = topic_difficulty.get(topic, {}).get("difficulty", 0)

   # Determine winner
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

   # Record proposition data
   if prop_initial is not None:
       prop_model = debate.proposition_model
       model_confidence_by_topic[prop_model][topic].append(prop_initial)
       model_baseline_confidence[prop_model].append(prop_initial)

       model_win_rates[prop_model]["total"] += 1
       if winner == "proposition":
           model_win_rates[prop_model]["wins"] += 1

   # Record opposition data
   if opp_initial is not None:
       opp_model = debate.opposition_model
       model_confidence_by_topic[opp_model][topic].append(opp_initial)
       model_baseline_confidence[opp_model].append(opp_initial)

       model_win_rates[opp_model]["total"] += 1
       if winner == "opposition":
           model_win_rates[opp_model]["wins"] += 1

# Calculate average confidence by model
model_avg_confidence = {model: sum(confidences)/len(confidences) if confidences else 0
                      for model, confidences in model_baseline_confidence.items()}

# Calculate win rates
for model in model_win_rates:
   if model_win_rates[model]["total"] > 0:
       model_win_rates[model]["rate"] = (model_win_rates[model]["wins"] /
                                        model_win_rates[model]["total"]) * 100
   else:
       model_win_rates[model]["rate"] = 0

# Calculate topic-adjusted confidence (sensitivity to topic difficulty)
model_topic_sensitivity = {}
for model, topic_data in model_confidence_by_topic.items():
   if not topic_data:
       continue

   # For each topic, calculate the average confidence
   topic_confidences = []
   topic_difficulties = []

   for topic, confidences in topic_data.items():
       if not confidences:
           continue

       avg_conf = sum(confidences) / len(confidences)
       topic_diff = topic_difficulty.get(topic, {}).get("difficulty", 0)

       topic_confidences.append(avg_conf)
       topic_difficulties.append(topic_diff)

   # Calculate correlation between topic difficulty and confidence
   if len(topic_confidences) >= 2:
       try:
           correlation = np.corrcoef(topic_difficulties, topic_confidences)[0, 1]
           model_topic_sensitivity[model] = correlation
       except:
           model_topic_sensitivity[model] = 0
   else:
       model_topic_sensitivity[model] = 0

# Print results
print("\n=== MODEL BASELINE CONFIDENCE ===")
for model, confidence in sorted(model_avg_confidence.items(), key=lambda x: x[1], reverse=True):
   if model in model_win_rates:
       win_rate = model_win_rates[model]["rate"]
       print(f"{model}: {confidence:.2f} average confidence, {win_rate:.1f}% win rate")

print("\n=== MODEL SENSITIVITY TO TOPIC DIFFICULTY ===")
print("(Positive values mean model is MORE confident on difficult topics)")
for model, sensitivity in sorted(model_topic_sensitivity.items(), key=lambda x: x[1], reverse=True):
   print(f"{model}: correlation {sensitivity:.2f}")

# Calculate overall correlation between baseline confidence and win rate
models_with_data = [m for m in model_avg_confidence.keys() if m in model_win_rates]
confidences = [model_avg_confidence[m] for m in models_with_data]
win_rates = [model_win_rates[m]["rate"] for m in models_with_data]

if len(confidences) >= 2:
   try:
       confidence_win_correlation = np.corrcoef(confidences, win_rates)[0, 1]
       print(f"\nCorrelation between model baseline confidence and win rate: {confidence_win_correlation:.2f}")

       if confidence_win_correlation > 0:
           print("Higher baseline confidence correlates with BETTER performance")
       else:
           print("Higher baseline confidence correlates with WORSE performance")
   except Exception:
       print("\nCouldn't calculate confidence-win correlation")
else:
   print("\nNot enough data to calculate confidence-win correlation")

print("\n=== CONFOUNDING FACTOR ANALYSIS ===")
print("If models have consistent confidence levels regardless of topic difficulty,")
print("but also have different win rates, this could explain your findings.")

print("\nAnalysis complete!")
