#!/usr/bin/env python3
"""
Script to count how many times each model was assigned to the proposition
and opposition sides in a set of debate tournaments.
"""

import logging
from pathlib import Path
from collections import defaultdict

from core.models import DebateTotal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("side_counts")

# Set the tournament directories
tournament_dirs = [
    Path("tournament/bet_tournament_20250316_1548"),
    Path("tournament/bet_tournament_20250317_1059")
]

# Initialize a dictionary to store the counts
model_side_counts = defaultdict(lambda: {"proposition": 0, "opposition": 0})

# Iterate through all tournament directories
for tournament_dir in tournament_dirs:
    logger.info(f"Processing tournament directory: {tournament_dir}")

    # Find all round directories within the tournament
    round_dirs = [d for d in tournament_dir.glob("round_*") if d.is_dir()]

    for round_dir in round_dirs:
        logger.info(f"  Processing round directory: {round_dir}")

        # Find all debate JSON files within the round
        for debate_path in round_dir.glob("*.json"):
            try:
                # Load the debate data
                debate = DebateTotal.load_from_json(debate_path)

                # Increment the counts for the proposition and opposition models
                model_side_counts[debate.proposition_model]["proposition"] += 1
                model_side_counts[debate.opposition_model]["opposition"] += 1

            except Exception as e:
                logger.error(f"Failed to load or process debate {debate_path}: {str(e)}")

# Print the results
print("\n=== Model Side Counts ===")
print("Model | Proposition | Opposition | Total | Prop/Opp Ratio")
print("------|-------------|------------|-------|---------------")

for model, counts in model_side_counts.items():
    total_assignments = counts["proposition"] + counts["opposition"]
    ratio = (counts["proposition"] / counts["opposition"]) if counts["opposition"] > 0 else "N/A (div by 0)"
    print(f"{model} | {counts['proposition']:11d} | {counts['opposition']:10d} | {total_assignments:5d} | {ratio}")

print("\nAnalysis Complete!")
