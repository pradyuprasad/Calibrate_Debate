#!/usr/bin/env python3
"""
Script to analyze if a model's confidence escalation relative to its average
escalation predicts the debate outcome, performing the analysis model by model
with statistical testing.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Any

from scipy.stats import fisher_exact # Import statistical tests

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
logger = logging.getLogger("model_by_model_escalation_predictiveness")

def analyze_model_specific_relative_escalation_predictiveness() -> None:
    """
    For each model, calculates its average confidence change, then for each
    instance of that model participating in a debate, calculates relative
    escalation and tests if positive relative escalation predicts winning
    using a model-specific statistical test.
    """
    logger.info("Loading debate data...")
    try:
        debate_results = load_debate_data()
        debates: List[DebateData] = debate_results.debates
        logger.info(f"Successfully loaded {len(debates)} debates.")
    except Exception as e:
        logger.error(f"Failed to load debate data: {e}")
        return

    # --- Step 1: Calculate overall average escalation per model ---
    model_total_changes: Dict[str, List[float]] = defaultdict(list)

    # First pass to collect all changes for each model
    for debate in debates:
        prop_opening_bet = debate.prop_bets.get(SpeechType.OPENING.value)
        prop_closing_bet = debate.prop_bets.get(SpeechType.CLOSING.value)
        opp_opening_bet = debate.opp_bets.get(SpeechType.OPENING.value)
        opp_closing_bet = debate.opp_bets.get(SpeechType.CLOSING.value)

        if prop_opening_bet is not None and prop_closing_bet is not None:
            model_total_changes[debate.proposition_model].append(prop_closing_bet - prop_opening_bet)

        if opp_opening_bet is not None and opp_closing_bet is not None:
            model_total_changes[debate.opposition_model].append(opp_closing_bet - opp_opening_bet)

    # Calculate the average change for each model
    model_overall_avg_escalation: Dict[str, float] = {}
    for model_name, changes in model_total_changes.items():
        if changes:
            model_overall_avg_escalation[model_name] = sum(changes) / len(changes)
        else:
             model_overall_avg_escalation[model_name] = 0.0

    logger.info("Calculated overall average escalation for models.")

    # --- Step 2 & 3: Calculate relative escalation and record outcome, grouped by model ---
    model_instance_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for debate in debates:
        prop_opening_bet = debate.prop_bets.get(SpeechType.OPENING.value)
        prop_closing_bet = debate.prop_bets.get(SpeechType.CLOSING.value)
        opp_opening_bet = debate.opp_bets.get(SpeechType.OPENING.value)
        opp_closing_bet = debate.opp_bets.get(SpeechType.CLOSING.value)

        # Process Proposition side
        if prop_opening_bet is not None and prop_closing_bet is not None:
            model_name = debate.proposition_model
            instance_change = prop_closing_bet - prop_opening_bet
            model_avg = model_overall_avg_escalation.get(model_name, 0.0) # Use 0.0 if model somehow not in overall stats
            relative_escalation = instance_change - model_avg
            model_instance_data[model_name].append({
                'relative_escalation': relative_escalation,
                'won': debate.winner == Side.PROPOSITION.value
            })

        # Process Opposition side
        if opp_opening_bet is not None and opp_closing_bet is not None:
            model_name = debate.opposition_model
            instance_change = opp_closing_bet - opp_opening_bet
            model_avg = model_overall_avg_escalation.get(model_name, 0.0) # Use 0.0 if model somehow not in overall stats
            relative_escalation = instance_change - model_avg
            model_instance_data[model_name].append({
                'relative_escalation': relative_escalation,
                'won': debate.winner == Side.OPPOSITION.value
            })

    # --- Step 4: Analyze win rates and perform statistical test for EACH MODEL ---

    print("\n" + "="*80)
    print("MODEL-BY-MODEL RELATIVE CONFIDENCE ESCALATION PREDICTIVENESS")
    print("Analysis: Does escalating MORE than THIS model's usual trend predict THIS model's win?")
    print("="*80)

    # Sort models alphabetically for consistent output
    sorted_models = sorted(model_instance_data.keys())

    if not sorted_models:
        print("No models found with complete betting data (opening and closing bets) for model-specific analysis.")
        return

    min_model_instances_for_test = 10 # Minimum total instances for a model to run the test
    min_instances_per_group = 2    # Minimum instances in positive AND negative groups for the test


    for model_name in sorted_models:
        instance_list = model_instance_data[model_name]

        print(f"\n--- Model: {model_name} ({len(instance_list)} instances with complete bets) ---")

        if not instance_list:
             print("  No data points with complete opening and closing bets for this model.")
             continue

        # Separate instances for this specific model
        positive_relative_escalation_instances = [
            inst for inst in instance_list if inst['relative_escalation'] > 0
        ]
        negative_relative_escalation_instances = [
            inst for inst in instance_list if inst['relative_escalation'] <= 0
        ]

        # Calculate counts for the contingency table for THIS MODEL
        total_positive = len(positive_relative_escalation_instances)
        wins_positive = sum(1 for inst in positive_relative_escalation_instances if inst['won'])
        losses_positive = total_positive - wins_positive

        total_negative = len(negative_relative_escalation_instances)
        wins_negative = sum(1 for inst in negative_relative_escalation_instances if inst['won'])
        losses_negative = total_negative - wins_negative

        win_rate_positive = (wins_positive / total_positive * 100) if total_positive > 0 else 0.0
        win_rate_negative = (wins_negative / total_negative * 100) if total_negative > 0 else 0.0

        print(f"  Instances with Positive Relative Escalation (> 0): {total_positive}")
        print(f"  Instances with Negative/Zero Relative Escalation (<= 0): {total_negative}")
        print(f"  Win Rate for Positive Relative Escalation instances: {win_rate_positive:.1f}%")
        print(f"  Win Rate for Negative/Zero Relative Escalation instances: {win_rate_negative:.1f}%")


        # --- Perform Model-Specific Statistical Test ---
        print("\n  --- Statistical Test (Relative Escalation vs. Outcome for THIS model) ---")

        # Construct the 2x2 contingency table for THIS MODEL
        # Rows: Relative Escalation Group (Positive, Negative/Zero)
        # Columns: Outcome (Won, Lost)
        contingency_table = [
            [wins_positive, losses_positive],   # Row 1: Positive Escalation instances
            [wins_negative, losses_negative]    # Row 2: Negative/Zero Escalation instances
        ]

        # Check if we have enough data *for this model* to perform the test
        # Need total instances AND instances in both groups
        if len(instance_list) < min_model_instances_for_test or total_positive < min_instances_per_group or total_negative < min_instances_per_group:
            print("  Insufficient data for statistical testing for this model.")
            print(f"  Requires at least {min_model_instances_for_test} total instances ({len(instance_list)} found) AND at least {min_instances_per_group} instances in both groups (Positive: {total_positive}, Negative/Zero: {total_negative}).")
            print("  Cannot perform Fisher's exact test for this model.")
        else:
            # Perform Fisher's exact test (robust for small counts in cells)
            try:
                # Use 'two-sided' alternative to see if there's *any* association
                # If you specifically want to test if Positive predicts Win (and Negative predicts Loss),
                # you might use 'greater' if the odds ratio > 1 is the effect you predict.
                odds_ratio, fisher_p_value = fisher_exact(contingency_table, alternative='two-sided')
                print("  Fisher's Exact Test:")
                print(f"    Contingency Table: {contingency_table}")
                print(f"    Odds Ratio: {odds_ratio:.3f}")
                print(f"    p-value: {fisher_p_value:.4f}")

                # Interpret Fisher's p-value
                alpha = 0.05
                if fisher_p_value < alpha:
                    print(f"    Result: Statistically significant difference detected (p < {alpha}) for THIS model.")
                    if win_rate_positive > win_rate_negative:
                         print("            For THIS model, instances with positive relative escalation are significantly more likely to win.")
                    else:
                         print("            For THIS model, instances with negative/zero relative escalation are significantly more likely to win.")
                else:
                    print(f"    Result: No statistically significant difference detected (p >= {alpha}) for THIS model.")
                    print("            For THIS model, relative escalation (relative to its average) does not significantly predict its winning.")

            except ValueError as e:
                print(f"  Could not perform Fisher's exact test for this model: {e}. Check if table has valid counts.")
                print(f"  Contingency table: {contingency_table}")

    print("\n" + "-"*40) # Separator for next model


    print("\n" + "="*80)
    print("Analysis Complete.")
    print("="*80)


if __name__ == "__main__":
    # To run this script, you need to have your debate data saved in the expected
    # location (e.g., 'private_bet_tournament' or wherever load_debate_data is configured).
    analyze_model_specific_relative_escalation_predictiveness()
