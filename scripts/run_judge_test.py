from pathlib import Path
import json
import re
from core.models import DebateTotal
from typing import Dict, List

def parse_judgment_files():
    """
    Parse all judgment files and organize results by debate and judge model.
    """
    # Get all relevant json files
    all_files = Path('.').glob('sample_debate_*.json')

    debates = {}  # debate_id -> original debate content
    judgments = {}  # debate_id -> {judge_model -> [runs]}

    for file in all_files:
        filename = file.name

        # Get base debate ID (like 'sample_debate_1')
        debate_id = 'sample_debate_' + filename.split('_')[2].split('.')[0]

        # Initialize judgments dict for this debate if needed
        if debate_id not in judgments:
            judgments[debate_id] = {}

        # Parse original debate files
        if '_judge_' not in filename:
            debate = DebateTotal.load_from_json(file)
            debates[debate_id] = debate
            continue

        # Parse judgment files
        parts = filename.replace('.json', '').split('_judge_')
        if len(parts) == 2:
            # Extract run number from end
            rest = parts[1].rsplit('_run_', 1)
            if len(rest) == 2:
                judge_model, run_num = rest

                # Load judgment file
                debate = DebateTotal.load_from_json(file)

                # Initialize dict for this judge if needed
                if judge_model not in judgments[debate_id]:
                    judgments[debate_id][judge_model] = []

                # Store judgment result
                if debate.judge_results:  # Check if there are any results
                    judgment = {
                        'run': int(run_num),
                        'winner': debate.judge_results[-1].winner,
                        'confidence': debate.judge_results[-1].confidence
                    }
                    judgments[debate_id][judge_model].append(judgment)

    return debates, judgments

def print_judgment_summary(judgments: Dict):
    """Print summary of collected judgments"""
    for debate_id in sorted(judgments.keys()):
        print(f"\nDebate: {debate_id}")
        for judge, runs in judgments[debate_id].items():
            if runs:  # Only print if there are actual judgments
                print(f"\nJudge: {judge}")
                for run in sorted(runs, key=lambda x: x['run']):
                    print(f"Run {run['run']}: Winner={run['winner']}, Confidence={run['confidence']}")

if __name__ == "__main__":
    debates, judgments = parse_judgment_files()
    print_judgment_summary(judgments)
