# generate_final_confidence_histogram.py

import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

# Import necessary components from your project structure
# Assuming these paths are correct relative to where you run this script
from scripts.analysis.load_data import load_debate_data
from scripts.analysis.models import DebateData # Assuming this is the item type in DebateResults.debates
from core.models import SpeechType # To reference the closing speech type


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define the directory where figures will be saved
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

def collect_final_confidences(debates_data_list: List[DebateData]) -> List[float]:
    """
    Collects all final (closing round) confidence bets from all debaters
    across all debates.

    Args:
        debates_data_list: List of standardized debate data objects.

    Returns:
        A list containing all final confidence bet amounts (floats).
    """
    final_confidences: List[float] = []
    closing_speech_key = SpeechType.CLOSING.value

    for debate_data in debates_data_list:
        # Get final confidence for Proposition
        prop_final_conf = debate_data.prop_bets.get(closing_speech_key, None)
        if prop_final_conf is not None:
             # Ensure the confidence value is a number before adding
            if isinstance(prop_final_conf, (int, float)):
                final_confidences.append(float(prop_final_conf)) # Ensure float type
            else:
                 logger.warning(f"Skipping non-numeric prop final bet: {prop_final_conf} for debate {debate_data.id}")


        # Get final confidence for Opposition
        opp_final_conf = debate_data.opp_bets.get(closing_speech_key, None)
        if opp_final_conf is not None:
             # Ensure the confidence value is a number before adding
            if isinstance(opp_final_conf, (int, float)):
                final_confidences.append(float(opp_final_conf)) # Ensure float type
            else:
                 logger.warning(f"Skipping non-numeric opp final bet: {opp_final_conf} for debate {debate_data.id}")


    return final_confidences

def generate_final_confidence_histogram(final_confidences_list: List[float]):
    """Generates a histogram of final confidence levels."""

    if not final_confidences_list:
        logger.warning("No final confidence data available to plot histogram.")
        return

    plt.figure(figsize=(10, 6)) # Adjust figure size as needed

    # Define bins for the histogram (e.g., bins of size 5 from 0 to 100)
    bins = range(0, 101, 5)

    plt.hist(final_confidences_list, bins=bins, edgecolor='black', alpha=0.7)

    plt.xlabel('Final Confidence (%)')
    plt.ylabel('Frequency (Number of Bets)')
    plt.title('Distribution of Final Stated Confidence Levels')
    plt.xlim(0, 100) # Set x-axis limits explicitly
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Optional: Add a vertical line at 50%
    # plt.axvline(x=50, color='red', linestyle='--', label='Expected Average Outcome (50%)')
    # plt.legend()


    plt.tight_layout() # Adjust layout

    figure_path = FIGURE_DIR / "final_confidence_histogram.pdf" # Or .png, .svg
    plt.savefig(figure_path)
    logger.info(f"Saved figure to {figure_path}")
    plt.close() # Close the figure to free memory


def main():
    """Main function to load data and generate the final confidence histogram."""
    logger.info("Loading debate data...")
    # load_debate_data should return an object with a list of standardized debate data
    debate_results = load_debate_data()
    debates_data_list: List[DebateData] = debate_results.debates

    logger.info(f"Loaded {len(debates_data_list)} debates.")

    # --- Collect final confidence data ---
    logger.info("Collecting final confidence levels...")
    final_confidences = collect_final_confidences(debates_data_list)
    logger.info(f"Collected {len(final_confidences)} final confidence values.")


    # --- Generate the Histogram Figure ---
    generate_final_confidence_histogram(final_confidences)

    logger.info("Figure generation complete.")

if __name__ == "__main__":
    main()
