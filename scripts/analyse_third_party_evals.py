import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

def load_results(filename: str = "debate_analysis_results.json") -> Dict:
    """Load the analysis results from the JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def calculate_difference(original: Optional[int], third_party: Optional[int]) -> Optional[float]:
    """Calculate the difference between original and third-party bets."""
    if original is None or third_party is None:
        return None
    return third_party - original

def analyze_by_debate(results: Dict) -> None:
    """Analyze differences by debate."""
    print("\n===== ANALYSIS BY DEBATE =====")

    for filename, debate in results.items():
        print(f"\nDebate: {filename}")
        print(f"Motion: {debate['motion']}")
        print(f"Proposition: {debate['proposition_model']}")
        print(f"Opposition: {debate['opposition_model']}")

        prop_diffs = []
        opp_diffs = []

        for stage_name in ['opening', 'rebuttal', 'closing']:
            stage_data = debate[stage_name]

            # Skip if data is incomplete
            if None in [stage_data['original_as_prop'], stage_data['third_party_as_prop'],
                       stage_data['original_as_opp'], stage_data['third_party_as_opp']]:
                continue

            prop_diff = calculate_difference(stage_data['original_as_prop'], stage_data['third_party_as_prop'])
            opp_diff = calculate_difference(stage_data['original_as_opp'], stage_data['third_party_as_opp'])

            if prop_diff is not None:
                prop_diffs.append(prop_diff)
            if opp_diff is not None:
                opp_diffs.append(opp_diff)

            print(f"  {stage_name.capitalize()}:")
            print(f"    Proposition: Original {stage_data['original_as_prop']} vs Third-party {stage_data['third_party_as_prop']} (Diff: {prop_diff:+d})")
            print(f"    Opposition:  Original {stage_data['original_as_opp']} vs Third-party {stage_data['third_party_as_opp']} (Diff: {opp_diff:+d})")

        # Calculate average differences
        if prop_diffs:
            avg_prop_diff = sum(prop_diffs) / len(prop_diffs)
            print(f"  Average Proposition Difference: {avg_prop_diff:+.2f}")

        if opp_diffs:
            avg_opp_diff = sum(opp_diffs) / len(opp_diffs)
            print(f"  Average Opposition Difference: {avg_opp_diff:+.2f}")

        if prop_diffs and opp_diffs:
            overall_diff = (sum(prop_diffs) + sum(opp_diffs)) / (len(prop_diffs) + len(opp_diffs))
            print(f"  Overall Average Difference: {overall_diff:+.2f}")

def analyze_by_model(results: Dict) -> None:
    """Analyze differences by model."""
    print("\n===== ANALYSIS BY MODEL =====")

    model_stats = defaultdict(lambda: {
        'as_prop': {'diffs': [], 'original': [], 'third_party': []},
        'as_opp': {'diffs': [], 'original': [], 'third_party': []},
    })

    # Collect data by model
    for filename, debate in results.items():
        prop_model = debate['proposition_model']
        opp_model = debate['opposition_model']

        for stage_name in ['opening', 'rebuttal', 'closing']:
            stage_data = debate[stage_name]

            # Skip if data is incomplete
            if None in [stage_data['original_as_prop'], stage_data['third_party_as_prop'],
                       stage_data['original_as_opp'], stage_data['third_party_as_opp']]:
                continue

            # Proposition model data
            prop_diff = calculate_difference(stage_data['original_as_prop'], stage_data['third_party_as_prop'])
            if prop_diff is not None:
                model_stats[prop_model]['as_prop']['diffs'].append(prop_diff)
                model_stats[prop_model]['as_prop']['original'].append(stage_data['original_as_prop'])
                model_stats[prop_model]['as_prop']['third_party'].append(stage_data['third_party_as_prop'])

            # Opposition model data
            opp_diff = calculate_difference(stage_data['original_as_opp'], stage_data['third_party_as_opp'])
            if opp_diff is not None:
                model_stats[opp_model]['as_opp']['diffs'].append(opp_diff)
                model_stats[opp_model]['as_opp']['original'].append(stage_data['original_as_opp'])
                model_stats[opp_model]['as_opp']['third_party'].append(stage_data['third_party_as_opp'])

    # Print results by model
    for model, stats in model_stats.items():
        print(f"\nModel: {model}")

        # As proposition
        prop_diffs = stats['as_prop']['diffs']
        if prop_diffs:
            avg_prop_diff = sum(prop_diffs) / len(prop_diffs)
            avg_original = sum(stats['as_prop']['original']) / len(stats['as_prop']['original'])
            avg_third_party = sum(stats['as_prop']['third_party']) / len(stats['as_prop']['third_party'])
            print(f"  As Proposition:")
            print(f"    Original avg: {avg_original:.2f}")
            print(f"    Third-party avg: {avg_third_party:.2f}")
            print(f"    Average difference: {avg_prop_diff:+.2f}")
        else:
            print("  No data as Proposition")

        # As opposition
        opp_diffs = stats['as_opp']['diffs']
        if opp_diffs:
            avg_opp_diff = sum(opp_diffs) / len(opp_diffs)
            avg_original = sum(stats['as_opp']['original']) / len(stats['as_opp']['original'])
            avg_third_party = sum(stats['as_opp']['third_party']) / len(stats['as_opp']['third_party'])
            print(f"  As Opposition:")
            print(f"    Original avg: {avg_original:.2f}")
            print(f"    Third-party avg: {avg_third_party:.2f}")
            print(f"    Average difference: {avg_opp_diff:+.2f}")
        else:
            print("  No data as Opposition")

        # Overall
        all_diffs = prop_diffs + opp_diffs
        if all_diffs:
            overall_diff = sum(all_diffs) / len(all_diffs)
            print(f"  Overall average difference: {overall_diff:+.2f}")
        else:
            print("  No overall data")

def analyze_overall_trends(results: Dict) -> None:
    """Analyze overall trends for proposition vs opposition."""
    print("\n===== OVERALL TRENDS =====")

    all_prop_original = []
    all_prop_third_party = []
    all_opp_original = []
    all_opp_third_party = []

    for filename, debate in results.items():
        for stage_name in ['opening', 'rebuttal', 'closing']:
            stage_data = debate[stage_name]

            if stage_data['original_as_prop'] is not None and stage_data['third_party_as_prop'] is not None:
                all_prop_original.append(stage_data['original_as_prop'])
                all_prop_third_party.append(stage_data['third_party_as_prop'])

            if stage_data['original_as_opp'] is not None and stage_data['third_party_as_opp'] is not None:
                all_opp_original.append(stage_data['original_as_opp'])
                all_opp_third_party.append(stage_data['third_party_as_opp'])

    # Proposition stats
    if all_prop_original and all_prop_third_party:
        avg_prop_original = sum(all_prop_original) / len(all_prop_original)
        avg_prop_third_party = sum(all_prop_third_party) / len(all_prop_third_party)
        avg_prop_diff = avg_prop_third_party - avg_prop_original
        print(f"Proposition:")
        print(f"  Original average: {avg_prop_original:.2f}")
        print(f"  Third-party average: {avg_prop_third_party:.2f}")
        print(f"  Average difference: {avg_prop_diff:+.2f}")

    # Opposition stats
    if all_opp_original and all_opp_third_party:
        avg_opp_original = sum(all_opp_original) / len(all_opp_original)
        avg_opp_third_party = sum(all_opp_third_party) / len(all_opp_third_party)
        avg_opp_diff = avg_opp_third_party - avg_opp_original
        print(f"Opposition:")
        print(f"  Original average: {avg_opp_original:.2f}")
        print(f"  Third-party average: {avg_opp_third_party:.2f}")
        print(f"  Average difference: {avg_opp_diff:+.2f}")

    # Overall stats (combining prop and opp)
    all_original = all_prop_original + all_opp_original
    all_third_party = all_prop_third_party + all_opp_third_party

    if all_original and all_third_party:
        avg_original = sum(all_original) / len(all_original)
        avg_third_party = sum(all_third_party) / len(all_third_party)
        avg_diff = avg_third_party - avg_original
        print(f"Overall:")
        print(f"  Original average: {avg_original:.2f}")
        print(f"  Third-party average: {avg_third_party:.2f}")
        print(f"  Average difference: {avg_diff:+.2f}")

def analyze_by_stage(results: Dict) -> None:
    """Analyze trends by debate stage."""
    print("\n===== ANALYSIS BY STAGE =====")

    stage_stats = {
        'opening': {'prop_diffs': [], 'opp_diffs': []},
        'rebuttal': {'prop_diffs': [], 'opp_diffs': []},
        'closing': {'prop_diffs': [], 'opp_diffs': []}
    }

    for filename, debate in results.items():
        for stage_name in ['opening', 'rebuttal', 'closing']:
            stage_data = debate[stage_name]

            prop_diff = calculate_difference(stage_data['original_as_prop'], stage_data['third_party_as_prop'])
            opp_diff = calculate_difference(stage_data['original_as_opp'], stage_data['third_party_as_opp'])

            if prop_diff is not None:
                stage_stats[stage_name]['prop_diffs'].append(prop_diff)
            if opp_diff is not None:
                stage_stats[stage_name]['opp_diffs'].append(opp_diff)

    for stage_name, stats in stage_stats.items():
        print(f"\n{stage_name.capitalize()} Stage:")

        prop_diffs = stats['prop_diffs']
        opp_diffs = stats['opp_diffs']

        if prop_diffs:
            avg_prop_diff = sum(prop_diffs) / len(prop_diffs)
            print(f"  Proposition average difference: {avg_prop_diff:+.2f}")

        if opp_diffs:
            avg_opp_diff = sum(opp_diffs) / len(opp_diffs)
            print(f"  Opposition average difference: {avg_opp_diff:+.2f}")

        all_diffs = prop_diffs + opp_diffs
        if all_diffs:
            overall_diff = sum(all_diffs) / len(all_diffs)
            print(f"  Overall average difference: {overall_diff:+.2f}")

def main():
    try:
        results = load_results()

        print("===== DEBATE ANALYSIS RESULTS =====")
        print(f"Total debates analyzed: {len(results)}")

        analyze_by_debate(results)
        analyze_by_model(results)
        analyze_overall_trends(results)
        analyze_by_stage(results)

    except FileNotFoundError:
        print("Error: debate_analysis_results.json not found. Please run the analysis script first.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in the results file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
