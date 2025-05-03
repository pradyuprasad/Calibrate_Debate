# generate_model_win_loss_confidence_dumbbell_arrow_plot.py

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches # Import for FancyArrowPatch

# Import necessary components from your project structure
# Assuming these paths are correct relative to where you run this script
from scripts.analysis.load_data import load_debate_data
from scripts.analysis.models import DebateData # Assuming this is the item type in DebateResults.debates
from core.models import Side, SpeechType # To reference sides and speech types

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define the directory where figures will be saved
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

def calculate_model_avg_win_loss_confidences(debates_data_list: List[DebateData]) -> Dict[str, Dict[str, float]]:
    """
    Calculates the average opening and closing confidence for each model,
    separated for debates won and lost by that model.

    Args:
        debates_data_list: List of standardized debate data objects.

    Returns:
        A dictionary where keys are model names and values are dictionaries
        containing 'win_opening', 'win_closing', 'loss_opening', 'loss_closing'
        average confidences. Returns NaN for categories with no data.
    """
    # Structure: model -> win/loss -> speech_type -> list of confidences
    model_confidences: defaultdict[str, defaultdict[str, defaultdict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for debate_data in debates_data_list:
        # Determine the models and outcome for this debate
        prop_model = debate_data.proposition_model
        opp_model = debate_data.opposition_model
        prop_won = debate_data.winner == Side.PROPOSITION.value

        # Process Proposition's bets
        outcome_key_prop = 'win' if prop_won else 'loss'
        for speech_type_enum in [SpeechType.OPENING, SpeechType.CLOSING]:
            speech_type_str = speech_type_enum.value
            confidence = debate_data.prop_bets.get(speech_type_str, None)
            if confidence is not None and isinstance(confidence, (int, float)):
                model_confidences[prop_model][outcome_key_prop][speech_type_str].append(float(confidence))
            elif confidence is not None:
                 logger.warning(f"Skipping non-numeric prop bet: {confidence} for debate {debate_data.id}, speech {speech_type_str}")


        # Process Opposition's bets
        outcome_key_opp = 'win' if not prop_won else 'loss' # Opp wins if Prop loses
        for speech_type_enum in [SpeechType.OPENING, SpeechType.CLOSING]:
            speech_type_str = speech_type_enum.value
            confidence = debate_data.opp_bets.get(speech_type_str, None)
            if confidence is not None and isinstance(confidence, (int, float)):
                 model_confidences[opp_model][outcome_key_opp][speech_type_str].append(float(confidence))
            elif confidence is not None:
                 logger.warning(f"Skipping non-numeric opp bet: {confidence} for debate {debate_data.id}, speech {speech_type_str}")


    # Calculate averages for each model, outcome, and speech type
    avg_model_confidences: Dict[str, Dict[str, float]] = {}
    for model, outcome_data in model_confidences.items():
        avg_model_confidences[model] = {}
        # Calculate averages for Winning instances
        for speech_type_str in [SpeechType.OPENING.value, SpeechType.CLOSING.value]:
             key = f'win_{speech_type_str}'
             if outcome_data['win'].get(speech_type_str): # Use .get() and check if list is non-empty
                 avg_model_confidences[model][key] = np.mean(outcome_data['win'][speech_type_str])
             else:
                 avg_model_confidences[model][key] = np.nan # NaN if no winning debates or no bet for that round

        # Calculate averages for Losing instances
        for speech_type_str in [SpeechType.OPENING.value, SpeechType.CLOSING.value]:
             key = f'loss_{speech_type_str}'
             if outcome_data['loss'].get(speech_type_str): # Use .get() and check if list is non-empty
                 avg_model_confidences[model][key] = np.mean(outcome_data['loss'][speech_type_str])
             else:
                 avg_model_confidences[model][key] = np.nan # NaN if no losing debates or no bet for that round


    return avg_model_confidences


def generate_model_win_loss_confidence_dumbbell_arrow_plot(avg_model_confs: Dict[str, Dict[str, float]]):
    """
    Generates a dumbbell plot with arrows showing average opening vs. closing confidence
    per model, separated for winning (solid green) and losing (dotted red) instances.
    """

    # Apply ggplot style
    plt.style.use('ggplot')

    # Prepare data for plotting
    # Each entry is (model, win_opening, win_closing, loss_opening, loss_closing)
    plot_data = []
    for model, confs in avg_model_confs.items():
        win_opening = confs.get('win_opening', np.nan)
        win_closing = confs.get('win_closing', np.nan)
        loss_opening = confs.get('loss_opening', np.nan)
        loss_closing = confs.get('loss_closing', np.nan)

        # Check if the model has data for at least one progression (win or loss)
        if (not np.isnan(win_opening) and not np.isnan(win_closing)) or \
           (not np.isnan(loss_opening) and not np.isnan(loss_closing)):
            plot_data.append((model, win_opening, win_closing, loss_opening, loss_closing))
        else:
             logger.info(f"Skipping model {model} from dumbbell plot due to insufficient win/loss confidence data.")


    if not plot_data:
        logger.warning("No models with complete win/loss confidence data to plot dumbbell chart after filtering.")
        # Revert style if nothing is plotted
        plt.style.use('default')
        return

    # Sort data by average CLOSING confidence WHEN WINNING (highest first = reverse=False for standard y-axis)
    # If a model has no winning data (win_closing is NaN), sort it to the bottom (-1).
    # If two models have NaN win_closing, their relative order doesn't strictly matter for the plot,
    # but key=lambda... will handle it consistently.
    sorted_plot_data = sorted(plot_data, key=lambda item: item[2] if not np.isnan(item[2]) else -1, reverse=False)

    models = [item[0] for item in sorted_plot_data]
    win_opening_confs = [item[1] for item in sorted_plot_data]
    win_closing_confs = [item[2] for item in sorted_plot_data]
    loss_opening_confs = [item[3] for item in sorted_plot_data]
    loss_closing_confs = [item[4] for item in sorted_plot_data]

    y_positions = np.arange(len(models)) # Y-axis positions for each model (0, 1, 2, ...)
    # Offset to place winning/losing dumbbells slightly above/below the main tick
    # Positive offset for Winning (above), negative offset for Losing (below)
    offset = 0.15

    # Determine figure height dynamically based on the number of models
    fig_height = max(6, len(models) * 0.8) # Adjust 0.8 for spacing
    fig, ax = plt.subplots(figsize=(12, fig_height)) # Create figure and axes objects

    # Arrow properties
    arrow_style = '-|>' # Line ending with an arrow
    arrow_mutation_scale = 15 # Adjust arrow head size
    arrow_lw = 3 # Line width

    # Text label offset
    text_offset_points = 10 # Horizontal offset

    # Plot Winning Dumbbells (Solid Green arrows, Green markers)
    for i in range(len(models)):
        # Only plot if winning data exists for this model
        if not np.isnan(win_opening_confs[i]) and not np.isnan(win_closing_confs[i]):
            y_pos = y_positions[i] + offset # Position for Winning row

            # Draw the arrow line from opening to closing confidence
            p = patches.FancyArrowPatch((win_opening_confs[i], y_pos), # Start point
                                        (win_closing_confs[i], y_pos), # End point
                                        arrowstyle=arrow_style,
                                        mutation_scale=arrow_mutation_scale,
                                        lw=arrow_lw,
                                        color='darkgreen',
                                        linestyle='-', # Solid line for Winning
                                        zorder=1) # Draw behind points
            ax.add_patch(p)

            # Plot opening confidence point (Green Circle)
            ax.scatter(win_opening_confs[i], y_pos, color='green', s=150, marker='o', label='Avg Opening Confidence (Win)', zorder=5)

            # Plot closing confidence point (Green Circle)
            ax.scatter(win_closing_confs[i], y_pos, color='forestgreen', s=150, marker='o', label='Avg Closing Confidence (Win)', zorder=5) # Still use darker shade for Closing

            # Add value labels using ax.annotate
            # Label for Opening Confidence (to the left)
            ax.annotate(f'{win_opening_confs[i]:.1f}%',
                        xy=(win_opening_confs[i], y_pos),
                        xytext=(-text_offset_points, 0), textcoords='offset points',
                        ha='right', va='center', fontsize=9, color='black',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))

            # Label for Closing Confidence (to the right)
            ax.annotate(f'{win_closing_confs[i]:.1f}%',
                        xy=(win_closing_confs[i], y_pos),
                        xytext=(text_offset_points, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=9, color='black',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))


    # Plot Losing Dumbbells (Dotted Red arrows, Red markers)
    for i in range(len(models)):
         # Only plot if losing data exists for this model
        if not np.isnan(loss_opening_confs[i]) and not np.isnan(loss_closing_confs[i]):
            y_pos = y_positions[i] - offset # Position for Losing row

            # Draw the arrow line from opening to closing confidence
            p = patches.FancyArrowPatch((loss_opening_confs[i], y_pos), # Start point
                                        (loss_closing_confs[i], y_pos), # End point
                                        arrowstyle=arrow_style,
                                        mutation_scale=arrow_mutation_scale,
                                        lw=arrow_lw,
                                        color='darkred',
                                        linestyle=':', # Dotted line for Losing
                                        zorder=1) # Draw behind points
            ax.add_patch(p)

            # Plot opening confidence point (Red Circle)
            ax.scatter(loss_opening_confs[i], y_pos, color='red', s=150, marker='o', label='Avg Opening Confidence (Loss)', zorder=5)

            # Plot closing confidence point (Red Circle)
            ax.scatter(loss_closing_confs[i], y_pos, color='orangered', s=150, marker='o', label='Avg Closing Confidence (Loss)', zorder=5) # Still use darker shade for Closing

            # Add value labels using ax.annotate
            text_offset_points = 10 # Horizontal offset

            # Label for Opening Confidence (to the left)
            ax.annotate(f'{loss_opening_confs[i]:.1f}%',
                        xy=(loss_opening_confs[i], y_pos),
                        xytext=(-text_offset_points, 0), textcoords='offset points',
                        ha='right', va='center', fontsize=9, color='black',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))

            # Label for Closing Confidence (to the right)
            ax.annotate(f'{loss_closing_confs[i]:.1f}%',
                        xy=(loss_closing_confs[i], y_pos),
                        xytext=(text_offset_points, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=9, color='black',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))


    # Add a vertical line for the 50% baseline
    ax.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, label='Expected Win Rate (50%)', zorder=0)

    # Set labels and title using axes methods
    ax.set_xlabel('Confidence (%)')
    ax.set_ylabel('Model')
    ax.set_title('Average Confidence Progression (Opening to Closing) per Model (Win vs. Loss)')
    ax.set_xlim(40, 100) # Set x-axis to start at 40%

    # Manually set x-ticks to ensure they are reasonable given the x-limit
    ax.set_xticks(np.arange(40, 101, 10)) # Ticks every 10 units from 40 to 100
    ax.set_ylim(-0.5, len(models) - 0.5) # Adjust y-limits to give space above/below first/last model

    ax.set_yticks(y_positions) # Set y-ticks at the positions where models are plotted
    ax.set_yticklabels(models) # Set model names as y-axis labels

    # Adjust grid visibility (ggplot style adds grids, customize if needed)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    # Turn off y-axis grid lines which can interfere with dumbbell plot clarity
    ax.grid(axis='y', visible=False)

    # Improve legend appearance
    # Manually create legend entries to ensure order and avoid duplicates from the loop
    legend_handles_revised = [
         plt.scatter([], [], color='green', s=150, marker='o'), # Win Opening/Closing Placeholder (same shape)
         plt.scatter([], [], color='red', s=150, marker='o'), # Loss Opening/Closing Placeholder (same shape)
         plt.Line2D([0], [0], color='darkgreen', lw=arrow_lw, linestyle='-'), # Win Trend Line Example
         plt.Line2D([0], [0], color='darkred', lw=arrow_lw, linestyle=':'), # Loss Trend Line Example (Dotted)
         plt.Line2D([0], [0], color='gray', linestyle='--', lw=1.5) # 50% line Placeholder
    ]
    legend_labels_revised = [
        'Avg Confidence (Win)', # Label for the points when winning
        'Avg Confidence (Loss)', # Label for the points when losing
        'Confidence Trend (Win)', # Label for the arrow line when winning
        'Confidence Trend (Loss)', # Label for the arrow line when losing
        'Expected Win Rate (50%)'
    ]


    ax.legend(legend_handles_revised, legend_labels_revised, loc='upper left', frameon=True, edgecolor='black')

    # Optional: Add text annotation to explain point color/shade mapping if needed
    # For example: "Green points/lines represent average confidence in debates won, Red in debates lost."
    # "Within Win/Loss: Opening = lighter shade circle, Closing = darker shade circle." # (This is not strictly true with current colors, maybe change marker colors?)

    # Let's simplify colors slightly for clarity:
    # Win: All points green, line darkgreen
    # Loss: All points red, line darkred
    # This matches the legend handles/labels better.
    # UPDATE SCATTER CALLS:
    # Win Opening: color='green', marker='o'
    # Win Closing: color='green', marker='o' (or forestgreen if you want distinct shades per point)
    # Loss Opening: color='red', marker='o'
    # Loss Closing: color='red', marker='o' (or orangered if you want distinct shades per point)
    # The code above already uses slightly different shades (green vs forestgreen, red vs orangered) which is good for seeing start/end points even with same shape. Let's keep that.


    fig.tight_layout() # Adjust layout

    figure_path = FIGURE_DIR / "model_win_loss_confidence_dumbbell_arrow_plot_ggplot.pdf" # Save with descriptive name
    plt.savefig(figure_path)
    logger.info(f"Saved figure to {figure_path}")

    # Revert style to default after plotting to not affect other plots
    plt.style.use('default')
    plt.close(fig) # Close the figure using the figure object


def main():
    """Main function to load data and generate the win/loss dumbbell plot."""
    logger.info("Loading debate data...")
    # load_debate_data should return an object with a list of standardized debate data
    # Filter out any debates that don't have debator_bets or judge_results as they are incomplete
    # and cannot be used for confidence analysis against outcome
    debate_results = load_debate_data()
    # Ensure we only consider debates with bet data AND judge results
    debates_data_list: List[DebateData] = [
        d for d in debate_results.debates
        if d.prop_bets is not None and d.opp_bets is not None and d.judge_results is not None
    ]

    logger.info(f"Loaded {len(debates_data_list)} debates with bet data and judge results.")

    # --- Calculate data for the dumbbell plot ---
    logger.info("Calculating average opening and closing confidence per model (Win vs Loss)...")
    avg_model_confs = calculate_model_avg_win_loss_confidences(debates_data_list)
    logger.info(f"Average Model Confidences (Win/Loss): {avg_model_confs}")

    # --- Generate the Dumbbell Plot Figure ---
    generate_model_win_loss_confidence_dumbbell_arrow_plot(avg_model_confs)

    logger.info("Figure generation complete.")

if __name__ == "__main__":
    main()
