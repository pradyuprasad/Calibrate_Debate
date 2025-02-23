from pathlib import Path
from collections import defaultdict
from typing import List, Dict
from core.models import DebateTotal

def load_round_debates(round_dir: Path) -> List[DebateTotal]:
   debates = []
   for debate_file in round_dir.glob("*.json"):
       if debate_file.name.endswith("_results.json"):
           continue
       debate = DebateTotal.load_from_json(debate_file)
       debates.append(debate)
   return debates

def process_debates():
   tournament_dir = Path("tournament")
   model_matches = defaultdict(int)

   for round_num in [1, 2, 3]:
       round_dir = tournament_dir / f"round_{round_num}"
       debates = load_round_debates(round_dir)

       print(f"\nRound {round_num} Judge Counts:")
       for debate in debates:
           prop_votes = sum(1 for result in debate.judge_results if result.winner == "proposition")
           print(f"{debate.proposition_model} vs {debate.opposition_model}: {prop_votes} prop votes out of {len(debate.judge_results)}")

           # Count matches for each model
           model_matches[debate.proposition_model] += 1
           model_matches[debate.opposition_model] += 1

   print("\nTotal matches per model:")
   for model, count in sorted(model_matches.items()):
       print(f"{model}: {count}")

if __name__ == "__main__":
   process_debates()
