# generate_model_confidence_bar_chart.py

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Import necessary components from your project structure
# Assuming these paths are correct relative to where you run this script
from scripts.analysis.load_data import load_debate_data
from scripts.analysis.models import DebateData # Assuming this is the item type in DebateResults.debates
from core.models import SpeechType # To reference the opening speech type


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define the directory where figures will be saved
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

def calculate_average_opening_confidence_per_model(debates_data_list: List[DebateData]) -> Dict[str, float]:
    """
    Calculates the average opening confidence bet for each model across all debates.

    Args:
        debates_data_list: List of standardized debate data objects.

    Returns:
        A dictionary with model names as keys and their average opening confidence as values.
    """
    model_opening_confidences: defaultdict[str, list[float]] = defaultdict(list)

    for debate_data in debates_data_list:
        # Get opening confidence for Proposition
        prop_conf = debate_data.prop_bets.get(SpeechType.OPENING.value, None)
        if prop_conf is not None:
            model_opening_confidences[debate_data.proposition_model].append(prop_conf)

        # Get opening confidence for Opposition
        opp_conf = debate_data.opp_bets.get(SpeechType.OPENING.value, None)
        if opp_conf is not None:
            model_opening_confidences[debate_data.opposition_model].append(opp_conf)

    # Calculate averages for each model
    avg_opening_confidence = {
        model: np.mean(confidences) if confidences else 0
        for model, confidences in model_opening_confidences.items()
    }

    return avg_opening_confidence

def generate_model_opening_confidence_bar_chart(model_avg_confs: Dict[str, float]):
    """Generates a bar chart showing average opening confidence per model."""

    # Sort models by average confidence (optional, but makes chart clearer)
    sorted_models = sorted(model_avg_confs.items(), key=lambda item: item[1], reverse=True)
    models = [item[0] for item in sorted_models]
    avg_confs = [item[1] for item in sorted_models]

    plt.figure(figsize=(12, 7)) # Adjust figure size as needed

    # Create horizontal bar chart
    plt.barh(models, avg_confs, color='skyblue')

    # Add a vertical line for the 50% expected win rate
    plt.axvline(x=50, color='red', linestyle='--', label='Expected Win Rate (50%)')

    plt.xlabel('Average Opening Confidence (%)')
    plt.ylabel('Model')
    plt.title('Average Opening Confidence per Model vs. Expected Win Rate')
    plt.xlim(0, 100) # Confidence is 0-100
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout() # Adjust layout to prevent labels overlapping

    figure_path = FIGURE_DIR / "model_avg_opening_confidence_bar_chart.pdf" # Or .png, .svg
    plt.savefig(figure_path)
    logger.info(f"Saved figure to {figure_path}")
    plt.close() # Close the figure to free memory

def main():
    """Main function to load data and generate the per-model confidence bar chart."""
    logger.info("Loading debate data...")
    # load_debate_data should return an object with a list of standardized debate data
    debate_results = load_debate_data()
    debates_data_list: List[DebateData] = debate_results.debates

    logger.info(f"Loaded {len(debates_data_list)} debates.")

    # --- Calculate data for the per-model bar chart ---
    logger.info("Calculating average opening confidence per model...")
    model_avg_opening_conf = calculate_average_opening_confidence_per_model(debates_data_list)
    logger.info(f"Per-model Avg Opening Confidence: {model_avg_opening_conf}")


    # --- Generate the Bar Chart Figure ---
    generate_model_opening_confidence_bar_chart(model_avg_opening_conf)

    logger.info("Figure generation complete.")

if __name__ == "__main__":
    main()
