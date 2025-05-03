#!/usr/bin/env python3
"""
Script to analyze average confidence escalation for models in debates,
separating results for winning and losing debates.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Assuming your project structure allows importing from core and scripts.analysis
# Adjust these imports if your structure is different.
try:
    from core.models import Side, SpeechType
    from scripts.analysis.load_data import load_debate_data
    from scripts.analysis.models import DebateData
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure you are running this script from the project root directory")
    print("or adjust the import paths accordingly.")
    exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("confidence_escalation_analysis")

def analyze_confidence_escalation_by_outcome() -> None:
    """
    Loads debate data and calculates the average confidence change (Closing - Opening)
    for each model, separated by whether the model won or lost the debate.
    Prints the results to the console.
    """
    logger.info("Loading debate data...")
    try:
        debate_results = load_debate_data()
        debates: List[DebateData] = debate_results.debates
        logger.info(f"Successfully loaded {len(debates)} debates.")
    except Exception as e:
        logger.error(f"Failed to load debate data: {e}")
        return

    # Data structure to store confidence changes per model, per outcome
    # model_name -> {'wins': [change1, change2, ...], 'losses': [changeA, changeB, ...]}
    model_changes_by_outcome: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {'wins': [], 'losses': []}
    )

    # Iterate through each debate
    for debate in debates:
        # Ensure we have opening and closing bets for both participants
        prop_opening_bet = debate.prop_bets.get(SpeechType.OPENING.value)
        prop_closing_bet = debate.prop_bets.get(SpeechType.CLOSING.value)
        opp_opening_bet = debate.opp_bets.get(SpeechType.OPENING.value)
        opp_closing_bet = debate.opp_bets.get(SpeechType.CLOSING.value)

        # Process Proposition side if bets are complete
        if prop_opening_bet is not None and prop_closing_bet is not None:
            prop_change = prop_closing_bet - prop_opening_bet
            if debate.winner == Side.PROPOSITION.value:
                model_changes_by_outcome[debate.proposition_model]['wins'].append(prop_change)
            else:
                model_changes_by_outcome[debate.proposition_model]['losses'].append(prop_change)

        # Process Opposition side if bets are complete
        if opp_opening_bet is not None and opp_closing_bet is not None:
            opp_change = opp_closing_bet - opp_opening_bet
            if debate.winner == Side.OPPOSITION.value:
                model_changes_by_outcome[debate.opposition_model]['wins'].append(opp_change)
            else:
                model_changes_by_outcome[debate.opposition_model]['losses'].append(opp_change)

    # Print analysis results per model
    print("\n" + "="*80)
    print("CONFIDENCE ESCALATION ANALYSIS BY DEBATE OUTCOME")
    print("="*80)

    # Sort models alphabetically for consistent output
    sorted_models = sorted(model_changes_by_outcome.keys())

    if not sorted_models:
        print("No models found with complete betting data (opening and closing bets).")
        return

    for model_name in sorted_models:
        changes = model_changes_by_outcome[model_name]
        win_changes = changes['wins']
        loss_changes = changes['losses']

        avg_win_change = sum(win_changes) / len(win_changes) if win_changes else None
        avg_loss_change = sum(loss_changes) / len(loss_changes) if loss_changes else None

        print(f"\n--- Model: {model_name} ---")
        print(f"  Average Confidence Change (Closing - Opening):")

        if avg_win_change is not None:
            print(f"    - In WINNING Debates ({len(win_changes)} instances): {avg_win_change:+.2f}")
        else:
            print(f"    - In WINNING Debates (0 instances): N/A (No wins with complete bet data)")

        if avg_loss_change is not None:
            print(f"    - In LOSING Debates ({len(loss_changes)} instances): {avg_loss_change:+.2f}")
        else:
            print(f"    - In LOSING Debates (0 instances): N/A (No losses with complete bet data)")

        # Optional: Comparison statement
        if avg_win_change is not None and avg_loss_change is not None:
            if avg_loss_change > avg_win_change:
                 print(f"  Comparison: Average escalation in LOSING debates ({avg_loss_change:+.2f}) is GREATER than in WINNING debates ({avg_win_change:+.2f}).")
            elif avg_loss_change < avg_win_change:
                 print(f"  Comparison: Average escalation in WINNING debates ({avg_win_change:+.2f}) is GREATER than in LOSING debates ({avg_loss_change:+.2f}).")
            else:
                 print(f"  Comparison: Average escalation is approximately equal in winning and losing debates ({avg_win_change:+.2f} vs {avg_loss_change:+.2f}).")
        elif avg_win_change is not None:
             print("  Comparison: Only winning debates with complete bet data found.")
        elif avg_loss_change is not None:
             print("  Comparison: Only losing debates with complete bet data found.")
        else:
             print("  Comparison: No complete bet data found for this model in any debate.")


    print("\n" + "="*80)
    print("Analysis Complete.")
    print("="*80)


if __name__ == "__main__":
    # To run this script, you need to have your debate data saved in the expected
    # location (e.g., 'private_bet_tournament' or wherever load_debate_data is configured).
    analyze_confidence_escalation_by_outcome()
