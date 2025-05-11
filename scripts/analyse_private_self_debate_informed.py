import os
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List

# Import necessary classes from core.models
from core.models import DebateTotal, Side, SpeechType

def load_debate_totals(directory_path: str) -> List[DebateTotal]:
    """
    Load all JSON files in the directory as DebateTotal objects.
    """
    debate_totals = []
    directory = Path(directory_path)

    for file_path in directory.glob("*.json"):
        try:
            debate_total = DebateTotal.load_from_json(file_path)
            debate_totals.append(debate_total)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return debate_totals

def extract_bet_data(debate_totals: List[DebateTotal]) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
    """
    Extract bet data from debate totals, organized by model, side, and speech type.
    Returns a nested dictionary: model -> side -> speech_type -> list of bet amounts
    """
    bet_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for debate in debate_totals:
        if not debate.debator_bets:
            continue

        # Both models are the same in self-debates
        model = debate.proposition_model

        for bet in debate.debator_bets:
            side = bet.side.value
            speech_type = bet.speech_type.value
            amount = bet.amount

            bet_data[model][side][speech_type].append(amount)

    return bet_data

def calculate_averages(bet_data: Dict) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate average bet amounts for each model, side, and speech type.
    """
    averages = {}

    for model, sides in bet_data.items():
        averages[model] = {}
        for side, speech_types in sides.items():
            averages[model][side] = {}
            for speech_type, amounts in speech_types.items():
                if amounts:
                    averages[model][side][speech_type] = sum(amounts) / len(amounts)
                else:
                    averages[model][side][speech_type] = 0.0

    return averages

def format_table(averages: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """
    Format the data as a table for terminal output.
    """
    # Define column headers and widths
    model_width = 40
    side_width = 6
    value_width = 10

    # Create header
    header = f"{'Model':{model_width}} | {'Side':{side_width}} | {'Opening':{value_width}} | {'Rebuttal':{value_width}} | {'Closing':{value_width}}"
    separator = "-" * (model_width + side_width + 3 * value_width + 8)

    # Create table
    lines = [header, separator]

    for model in sorted(averages.keys()):
        prop_opening = averages[model].get('proposition', {}).get('opening', 0.0)
        prop_rebuttal = averages[model].get('proposition', {}).get('rebuttal', 0.0)
        prop_closing = averages[model].get('proposition', {}).get('closing', 0.0)

        opp_opening = averages[model].get('opposition', {}).get('opening', 0.0)
        opp_rebuttal = averages[model].get('opposition', {}).get('rebuttal', 0.0)
        opp_closing = averages[model].get('opposition', {}).get('closing', 0.0)

        # Add proposition line
        lines.append(f"{model:{model_width}} | {'Prop':{side_width}} | {prop_opening:10.1f} | {prop_rebuttal:10.1f} | {prop_closing:10.1f}")

        # Add opposition line
        lines.append(f"{'':{model_width}} | {'Opp':{side_width}} | {opp_opening:10.1f} | {opp_rebuttal:10.1f} | {opp_closing:10.1f}")

        # Add separator between models
        lines.append(separator)

    return "\n".join(lines)

def count_debates_per_model(debate_totals: List[DebateTotal]) -> Dict[str, int]:
    """
    Count how many debates each model participated in
    """
    model_debates = Counter()

    for debate in debate_totals:
        model = debate.proposition_model  # In self-debates, prop and opp are the same model
        model_debates[model] += 1

    return model_debates

def main():
    directory_path = "private_self_debates_informed"  # Path to the directory containing the JSON files

    # Load all debate totals
    debate_totals = load_debate_totals(directory_path)
    total_debates = len(debate_totals)

    # 1. Print each model's debate count
    model_debate_counts = count_debates_per_model(debate_totals)
    print("1. DEBATES PER MODEL:")
    for model, count in sorted(model_debate_counts.items()):
        print(f"{model}: {count} debates")

    # 2. Print total number of debates
    print(f"\n2. TOTAL DEBATES: {total_debates}")

    # 3. Print the betting confidence table
    print("\n3. MODEL BETTING CONFIDENCE TABLE:")
    bet_data = extract_bet_data(debate_totals)
    averages = calculate_averages(bet_data)
    table = format_table(averages)
    print(table)

if __name__ == "__main__":
    main()
