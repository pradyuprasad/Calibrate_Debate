'''
1. load models
'''
from ai_models.load_models import load_debate_models
from typing import List, Set, Tuple, Literal
from core.models import JudgeResult, Match, DebatePrompts, DebateTopic, DebateTotal, ConfidenceAdjustedJudgement
from topics.load_topics import load_topics
from config import Config
from core.debate import run_debate, get_judgement
import random
from scripts.continue_debate import continue_debate
from pathlib import Path
from itertools import cycle, islice
from collections import deque
from scripts.utils import checkIfComplete, sanitize_model_name
from prompts.load_prompts import get_debate_prompt
import json
import logging


def create_new_matches(models_list: List[str], previous_matches:Set[Match]) -> List[Match]:
    random.shuffle(models_list)

    paired = set()
    output = []
    for i in range(len(models_list)):
        if models_list[i] not in paired:
            for j in range(len(models_list)):
                if j == i:
                    continue
                possible_match = Match(prop_model=models_list[i], opp_model=models_list[j])
                if models_list[j] not in paired and possible_match not in previous_matches:
                    output.append(possible_match)
                    paired.add(models_list[i])
                    paired.add(models_list[j])
                    previous_matches.add(possible_match)

    return output

def get_topics(config: Config, num_topics: int) -> List[DebateTopic]:
    topics = load_topics(config=config)
    random.shuffle(topics)

    return list(islice(cycle(topics), num_topics))



def process_failed_debates(failed_debate_queue: deque[Path]) -> None:
    if not failed_debate_queue:
        logging.error("No failed debate queue")
        return

    still_failed_queue : deque[Path] = deque()

    while failed_debate_queue:  # Process until the queue is empty
        debate_path = failed_debate_queue.popleft()  # Get the first item in the queue

        if not debate_path.exists():
            logging.error(f"{debate_path} does not exist")
            continue

        try:
            continue_debate(debate_path=debate_path)
        except Exception as e:
            print(f"Debate failed again with error: {str(e)}")
            still_failed_queue.append(debate_path)  # Re-add to the end of the queue for retry

    if still_failed_queue:
        print(f"\n{len(still_failed_queue)} debates still failed after retry.")
    else:
        print("\nAll failed debates were successfully processed.")


def count_prop_winners(judgements: List[JudgeResult]) -> int:
    prop_winners = 0
    for judgement in judgements:
        if judgement.winner == 'proposition':
            prop_winners += 1

    return prop_winners

def get_repeated_judgements(debate: DebateTotal, prompts: DebatePrompts, judge_model: str, num_judgements:int) -> List[JudgeResult]:
    output = []
    for _ in range(num_judgements):
        output.append(get_judgement(debate, prompts, judge_model))

    return output

def judge_single_debate(debate_path: Path, prompts: DebatePrompts) -> Tuple[Literal['opposition', 'proposition'], float]:

    if not checkIfComplete(debate_path):
        raise Exception(f"{debate_path} is not complete")

    debate = DebateTotal.load_from_json(debate_path)

    judgements = get_repeated_judgements(debate, prompts, "deepseek/deepseek-chat", num_judgements=3)

    prop_winners_first = count_prop_winners(judgements=judgements)
    op_winners_first = len(judgements) - prop_winners_first
    winner: Literal['opposition', 'proposition']
    if prop_winners_first == len(judgements):
        winner = 'proposition'
        return (winner, 1.0)
    elif prop_winners_first == 0:
        winner = 'opposition'
        return (winner, 1.0)
    o1_judgements = get_repeated_judgements(
        debate=debate,
        prompts=prompts,
        judge_model="openai/o1-mini",
        num_judgements=3
    )

    prop_winners_second = count_prop_winners(o1_judgements)
    op_winners_second= len(o1_judgements) - prop_winners_second
    if prop_winners_second >= 2:
        winner = 'proposition'
        margin = (prop_winners_second + prop_winners_first) / (len(judgements) + len(o1_judgements))

    else:
        winner = 'opposition'
        margin = (op_winners_first + op_winners_second) / (len(judgements) + len(o1_judgements))

    return winner, margin



def judge_single_round(round_dir: Path, judgements_stored_path: Path, prompts: DebatePrompts) -> None:
    path_list = list(round_dir.glob("*.json"))
    output_list = []
    for path in path_list:
        debate = DebateTotal.load_from_json(path)
        winner, margin = judge_single_debate(path, prompts)
        judgement_total = ConfidenceAdjustedJudgement(
            prop_model=debate.proposition_model, opp_model=debate.opposition_model,
            winner=winner,
            margin=margin
        )
        output_list.append(judgement_total)

        with open(judgements_stored_path, 'w') as f:
            json.dump(output_list, f)

def run_single_round(config:Config, models_list: List[str], previous_matches:Set[Match], prompts: DebatePrompts, dir_to_store: Path) -> None:
    failed_debate_queue: deque[Path] = deque()
    new_matches = create_new_matches(models_list=models_list, previous_matches=previous_matches)
    topic_list = get_topics(config=config, num_topics=len(new_matches))
    assert len(new_matches) == len(topic_list), f"Expected same length for new_matches and topic_list, but got new_matches length as {len(new_matches)} and topic_list length as {len(topic_list)}"
    for match, topic in zip(new_matches, topic_list):
        prop_model_name_clean = sanitize_model_name(match.prop_model)
        opp_model_name_clean = sanitize_model_name(match.opp_model)
        path_to_store = dir_to_store / f"{prop_model_name_clean}_vs_{opp_model_name_clean}.json"
        logging.info(f"Going to run {path_to_store}")
        try:
            run_debate(proposition_model=match.prop_model,
                    opposition_model=match.opp_model,
                    motion=topic,
                    prompts=prompts,
                    path_to_store_debate=path_to_store,
                    judge_models=[])

        except Exception as e:
            logging.error(f"Error {e}")
            failed_debate_queue.append(path_to_store)

    process_failed_debates(failed_debate_queue=failed_debate_queue)




def main():
    config = Config()
    models_list = list(load_debate_models(config).keys())
    prev_matches = set()
    prompts = get_debate_prompt(config)
    for i in range(1, 4):
        round_path = Path(config.tournament_dir / f"round_{i}")
        round_path.mkdir(exist_ok=True)
        run_single_round(config=config, models_list=models_list, previous_matches=prev_matches, prompts=prompts, dir_to_store=round_path)

main()
