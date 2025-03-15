from pathlib import Path
from typing import List, Set, Deque, Tuple, Literal
from collections import deque
import logging
import random
import json
from itertools import cycle, islice
from dotenv import load_dotenv
import os

from core.models import (
    Match,
    DebatePrompts,
    DebateTopic,
    DebateTotal,
    ConfidenceAdjustedJudgement,
    JudgeResult,
)
from core.api_client import OpenRouterClient
from core.message_formatter import MessageFormatter
from core.debate_service import DebateService
from core.judgement_processor import JudgementProcessor
from ai_models.load_models import load_debate_models
from topics.load_topics import load_topics
from prompts.load_prompts import get_debate_prompt
from config import Config
from scripts.utils import sanitize_model_name, checkIfComplete

logger = logging.getLogger(__name__)
load_dotenv()


class JudgingService:
    def __init__(
        self,
        judgement_processor: JudgementProcessor,
        primary_judge_model: str = "deepseek/deepseek-chat",
        secondary_judge_model: str = "openai/o1-mini",
        judgements_per_model: int = 3,
    ):
        self.judgement_processor = judgement_processor
        self.primary_judge = primary_judge_model
        self.secondary_judge = secondary_judge_model
        self.judgements_per_model = judgements_per_model
        self.logger = logging.getLogger(__name__)

    def get_repeated_judgements(
        self, debate: DebateTotal, judge_model: str
    ) -> List[JudgeResult]:
        """Get multiple judgements from the same model"""
        judgements = []
        for _ in range(self.judgements_per_model):
            try:
                judgement = self.judgement_processor.process_judgment(
                    debate=debate, model=judge_model
                )
                judgements.append(judgement)
            except Exception as e:
                self.logger.error(f"Failed to get judgement: {e}")
                continue
        return judgements

    def calculate_winner_confidence(
        self, judgements: List[JudgeResult]
    ) -> Tuple[Literal["opposition", "proposition"], float]:
        """Calculate winner and confidence from a list of judgements"""
        if not judgements:
            raise ValueError("No valid judgements to process")

        prop_wins = 0
        total_confidence = 0

        for judgement in judgements:
            if judgement.winner == "proposition":
                prop_wins += 1
                total_confidence += judgement.confidence
            else:
                total_confidence -= judgement.confidence

        avg_confidence = abs(total_confidence) / len(judgements)
        normalized_confidence = min(avg_confidence / 100, 1.0)

        if prop_wins > len(judgements) / 2:
            return ("proposition", normalized_confidence)
        else:
            return ("opposition", normalized_confidence)

    def judge_debate(
        self, debate_path: Path
    ) -> Tuple[Literal["opposition", "proposition"], float]:
        """Judge a single debate with multiple rounds if needed"""
        if not checkIfComplete(debate_path):
            raise ValueError(f"{debate_path} is not complete")

        debate = DebateTotal.load_from_json(debate_path)

        # First round of judgements
        primary_judgements = self.get_repeated_judgements(
            debate=debate, judge_model=self.primary_judge
        )

        if not primary_judgements:
            raise ValueError("Failed to get any primary judgements")

        winner, confidence = self.calculate_winner_confidence(primary_judgements)

        # If confidence is high, return result
        if confidence > 0.8:
            return winner, confidence

        # Get secondary judgements for close calls
        secondary_judgements = self.get_repeated_judgements(
            debate=debate, judge_model=self.secondary_judge
        )

        # Combine all judgements for final decision
        all_judgements = primary_judgements + secondary_judgements
        return self.calculate_winner_confidence(all_judgements)


class TournamentService:
    def __init__(
        self,
        debate_service: DebateService,
        judging_service: JudgingService,
        config: Config,
        prompts: DebatePrompts,
    ):
        self.debate_service = debate_service
        self.judging_service = judging_service
        self.config = config
        self.prompts = prompts
        self.failed_debates: Deque[Path] = deque()

    def create_new_matches(
        self, models_list: List[str], previous_matches: Set[Match]
    ) -> List[Match]:
        random.shuffle(models_list)
        paired = set()
        output = []

        for i, model_i in enumerate(models_list):
            if model_i not in paired:
                for j, model_j in enumerate(models_list):
                    if j == i:
                        continue
                    possible_match = Match(prop_model=model_i, opp_model=model_j)
                    if (
                        model_j not in paired
                        and possible_match not in previous_matches
                        and model_i not in paired
                    ):
                        output.append(possible_match)
                        paired.add(model_i)
                        paired.add(model_j)
                        previous_matches.add(possible_match)

        return output

    def get_topics(self, num_topics: int) -> List[DebateTopic]:
        topics = load_topics(config=self.config)
        random.shuffle(topics)
        return list(islice(cycle(topics), num_topics))

    def process_failed_debates(self) -> None:
        still_failed: Deque[Path] = deque()

        while self.failed_debates:
            debate_path = self.failed_debates.popleft()

            if not debate_path.exists():
                logger.error(f"{debate_path} does not exist")
                continue

            try:
                self.debate_service.continue_debate(debate_path)
            except Exception as e:
                logger.error(f"Debate failed again: {str(e)}")
                still_failed.append(debate_path)

        if still_failed:
            logger.error(f"{len(still_failed)} debates still failed after retry")
        else:
            logger.info("All failed debates were successfully processed")

    def save_round_results(
        self, round_num: int, judgements: List[ConfidenceAdjustedJudgement]
    ) -> None:
        results_path = self.config.tournament_dir / f"round_{round_num}_results.json"
        with open(results_path, "w") as f:
            json.dump([j.dict() for j in judgements], f, indent=2)

    def run_single_round(
        self,
        round_num: int,
        models_list: List[str],
        previous_matches: Set[Match],
        dir_to_store: Path,
    ) -> None:
        logger.info(f"Starting round {round_num}")

        # Run debates
        new_matches = self.create_new_matches(models_list, previous_matches)
        topic_list = self.get_topics(num_topics=len(new_matches))

        assert len(new_matches) == len(topic_list), (
            f"Mismatch in matches ({len(new_matches)}) and topics ({len(topic_list)})"
        )

        round_judgements = []
        for match, topic in zip(new_matches, topic_list):
            prop_name = sanitize_model_name(match.prop_model)
            opp_name = sanitize_model_name(match.opp_model)
            debate_path = dir_to_store / f"{prop_name}_vs_{opp_name}.json"

            try:
                # Run debate
                logger.info(f"Running debate: {debate_path}")
                self.debate_service.run_debate(
                    proposition_model=match.prop_model,
                    opposition_model=match.opp_model,
                    motion=topic,
                    path_to_store=debate_path,
                )

                # Judge debate
                logger.info(f"Judging debate: {debate_path}")
                winner, margin = self.judging_service.judge_debate(debate_path)

                # Store result
                judgement = ConfidenceAdjustedJudgement(
                    prop_model=match.prop_model,
                    opp_model=match.opp_model,
                    winner=winner,
                    margin=margin,
                )
                round_judgements.append(judgement)

            except Exception as e:
                logger.error(f"Error in debate: {e}")
                self.failed_debates.append(debate_path)

        # Process any failed debates
        self.process_failed_debates()

        # Save round results
        self.save_round_results(round_num, round_judgements)

    def run_tournament(self, num_rounds: int = 3) -> None:
        models_list = list(load_debate_models(self.config).keys())
        previous_matches: Set[Match] = set()

        for round_num in range(1, num_rounds + 1):
            round_path = self.config.tournament_dir / f"round_{round_num}"
            round_path.mkdir(exist_ok=True)

            self.run_single_round(
                round_num=round_num,
                models_list=models_list,
                previous_matches=previous_matches,
                dir_to_store=round_path,
            )


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize dependencies
    config = Config()
    prompts = get_debate_prompt(config)
    api_client = OpenRouterClient(os.environ.get("OPENROUTER_API_KEY"))
    message_formatter = MessageFormatter(prompts)

    # Create services
    judgement_processor = JudgementProcessor(prompts, api_client)
    judging_service = JudgingService(judgement_processor)
    debate_service = DebateService(api_client, message_formatter)

    # Create and run tournament
    tournament = TournamentService(debate_service, judging_service, config, prompts)
    tournament.run_tournament()


if __name__ == "__main__":
    main()
