# generate_side_comparison_figure.py

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Import necessary components from your project structure
# Assuming these paths are correct relative to where you run this script
from scripts.analysis.load_data import load_debate_data
from scripts.analysis.models import DebateData # Assuming this is the item type in DebateResults.debates
from core.models import Side # To reference sides and speech types


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define the directory where figures will be saved
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

def calculate_side_metrics(debates_data_list: List[DebateData]) -> Dict[str, Dict[str, float]]:
    """
    Calculates the overall win rate and average confidence for Proposition and Opposition sides.
    Average confidence is calculated across all rounds for each side.

    Args:
        debates_data_list: List of standardized debate data objects.

    Returns:
        A dictionary with side names ("proposition", "opposition") as keys,
        each containing a nested dictionary with "win_rate" and "avg_confidence".
    """
    prop_wins = 0
    prop_debates_count = 0
    prop_total_confidence = 0
    prop_bet_count = 0

    opp_wins = 0
    opp_debates_count = 0
    opp_total_confidence = 0
    opp_bet_count = 0

    for debate_data in debates_data_list:
        # --- Metrics for Proposition Side ---
        prop_debates_count += 1 # Prop participates in every debate as prop
        if debate_data.winner == Side.PROPOSITION.value:
            prop_wins += 1

        for speech_type_str, confidence in debate_data.prop_bets.items():
             # Ensure the confidence value is a number before adding
            if isinstance(confidence, (int, float)):
                prop_total_confidence += confidence
                prop_bet_count += 1
            else:
                 logger.warning(f"Skipping non-numeric prop bet: {confidence} for debate {debate_data.id}, speech {speech_type_str}")


        # --- Metrics for Opposition Side ---
        opp_debates_count += 1 # Opp participates in every debate as opp
        if debate_data.winner == Side.OPPOSITION.value:
            opp_wins += 1

        for speech_type_str, confidence in debate_data.opp_bets.items():
            # Ensure the confidence value is a number before adding
            if isinstance(confidence, (int, float)):
                opp_total_confidence += confidence
                opp_bet_count += 1
            else:
                 logger.warning(f"Skipping non-numeric opp bet: {confidence} for debate {debate_data.id}, speech {speech_type_str}")


    # Calculate final metrics
    prop_win_rate = (prop_wins / prop_debates_count * 100) if prop_debates_count > 0 else 0
    prop_avg_confidence = (prop_total_confidence / prop_bet_count) if prop_bet_count > 0 else 0

    opp_win_rate = (opp_wins / opp_debates_count * 100) if opp_debates_count > 0 else 0
    opp_avg_confidence = (opp_total_confidence / opp_bet_count) if opp_bet_count > 0 else 0

    return {
        Side.PROPOSITION.value: {"win_rate": prop_win_rate, "avg_confidence": prop_avg_confidence},
        Side.OPPOSITION.value: {"win_rate": opp_win_rate, "avg_confidence": opp_avg_confidence}
    }

def generate_side_comparison_figure(side_metrics: Dict[str, Dict[str, float]]):
    """Generates a grouped bar chart comparing Win Rate and Average Confidence for sides."""

    labels = [Side.PROPOSITION.value.capitalize(), Side.OPPOSITION.value.capitalize()]
    win_rates = [side_metrics[Side.PROPOSITION.value]["win_rate"], side_metrics[Side.OPPOSITION.value]["win_rate"]]
    avg_confidences = [side_metrics[Side.PROPOSITION.value]["avg_confidence"], side_metrics[Side.OPPOSITION.value]["avg_confidence"]]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 6)) # Adjust figure size as needed

    rects1 = ax.bar(x - width/2, win_rates, width, label='Actual Win Rate')
    rects2 = ax.bar(x + width/2, avg_confidences, width, label='Average Stated Confidence')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Comparison of Win Rate and Average Stated Confidence by Side')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100) # Percentages are 0-100
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=0.8, label='Expected Win Rate (50%)') # Add 50% line

    # Add the value labels on top of the bars
    def autolabel(rects):
        """Attach a text label above each bar in *rects*"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    # Handle legend placement
    ax.legend(loc='upper left')

    fig.tight_layout() # Adjust layout to prevent labels overlapping

    figure_path = FIGURE_DIR / "side_winrate_confidence_comparison.pdf" # Or .png, .svg
    plt.savefig(figure_path)
    logger.info(f"Saved figure to {figure_path}")
    plt.close(fig) # Close the figure to free memory


def main():
    """Main function to load data and generate the side comparison figure."""
    logger.info("Loading debate data...")
    # load_debate_data should return an object with a list of standardized debate data
    debate_results = load_debate_data()
    debates_data_list: List[DebateData] = debate_results.debates

    logger.info(f"Loaded {len(debates_data_list)} debates.")

    # --- Calculate data for the side comparison figure ---
    logger.info("Calculating side-specific win rates and average confidences...")
    side_metrics = calculate_side_metrics(debates_data_list)
    logger.info(f"Side Metrics: {side_metrics}")


    # --- Generate the Figure ---
    generate_side_comparison_figure(side_metrics)

    logger.info("Figure generation complete.")

if __name__ == "__main__":
    main()
