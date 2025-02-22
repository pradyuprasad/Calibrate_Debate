from pathlib import Path
import json
import re
from core.models import DebateTotal
from typing import Dict, List
from config import Config

def parse_judgment_files():
    """
    Parse all judgment files and organize results by debate and judge model.
    """
    config = Config()

    # Get all relevant json files from debates and judgments directories
    debate_files = list(config.sample_debates_dir.glob('sample_debate_*.json'))
    judgment_files = list(config.sample_judgments_dir.glob('sample_debate_*_judge_*.json'))

    debates = {}  # debate_id -> original debate content
    judgments = {}  # debate_id -> {judge_model -> [runs]}

    # Process original debate files
    for file in debate_files:
        debate_id = file.stem  # Gets filename without extension
        debate = DebateTotal.load_from_json(file)
        debates[debate_id] = debate
        judgments[debate_id] = {}

    # Process judgment files
    for file in judgment_files:
        filename = file.stem
        parts = filename.split('_judge_')
        if len(parts) == 2:
            debate_id = parts[0]
            rest = parts[1].rsplit('_run_', 1)
            if len(rest) == 2:
                judge_model, run_num = rest

                # Initialize dict for this judge if needed
                if debate_id not in judgments:
                    judgments[debate_id] = {}
                if judge_model not in judgments[debate_id]:
                    judgments[debate_id][judge_model] = []

                # Load and store judgment result
                debate = DebateTotal.load_from_json(file)
                if debate.judge_results:
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
