#!/usr/bin/env python3
"""
Script to run debates based on the balanced schedule recommendations,
saving each debate individually with timestamps. Includes judging.
"""

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv

from config import Config
from core.models import DebateTopic, DebateType, JudgeResult
from scripts.utils import checkIfComplete, sanitize_model_name
from topics.load_topics import load_topics


def setup_logging(config: Config):
    """Configure logging for the debate runner."""
    logger = config.logger.get_logger()
    return logger

class BalancedDebateRunner:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)

        # Create output directory if it doesn't exist
        self.output_dir = Path("private_bet_tournament")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load topics
        self.topics = load_topics(config)
        self.logger.info(f"Loaded {len(self.topics)} debate topics")

        # Configure judge models
        self.judge_models = [
            "qwen/qwq-32b",
            "google/gemini-pro-1.5",
            "deepseek/deepseek-chat",
        ]
        self.voting_rounds = 2

        self.logger.info(
            f"Using {len(self.judge_models)} judges for evaluation: {', '.join(self.judge_models)}"
        )

        # Try to load the debate pairings
        try:
            with open("balance_debate.json", "r") as f:
                self.debate_pairings = json.load(f)
            self.logger.info(f"Loaded {len(self.debate_pairings)} debate pairings from balance_debate.json")
        except FileNotFoundError:
            self.logger.error("balance_debate.json not found. Run the recommendation script first.")
            raise

    def run_debate(self, match: Dict, topic: DebateTopic, output_path: Path) -> bool:
        """Run a debate between two models with private betting."""
        prop_model = match["proposition"]
        opp_model = match["opposition"]

        self.logger.info(f"Starting debate: {prop_model} vs {opp_model}")
        self.logger.info(f"Topic: {topic.topic_description}")

        try:
            debate = self.config.debate_service.run_debate(
                proposition_model=prop_model,
                opposition_model=opp_model,
                motion=topic,
                path_to_store=output_path,
                debate_type=DebateType.PRIVATE_BET,
            )
            self.logger.info(f"Debate completed successfully and saved to {output_path}")

            # Log bet information if available
            if debate.debator_bets:
                self.logger.info("Bets placed:")
                for bet in debate.debator_bets:
                    self.logger.info(f"{bet.side.value} {bet.speech_type.value}: {bet.amount}")

            return True
        except Exception as e:
            self.logger.error(f"Debate failed: {str(e)}")
            return False

    def judge_debate(self, debate_path: Path) -> Optional[List[JudgeResult]]:
        """Judge a debate and return the judgments using all available judge models."""
        self.logger.info(f"Judging debate: {debate_path}")

        try:
            debate = self.config.debate_service.continue_debate(debate_path)
            all_judgements: List[JudgeResult] = []

            for i in range(self.voting_rounds):
                # Get judgment from each judge model
                for judge_model in self.judge_models:
                    judgment = self.config.judgement_processor.process_judgment(
                        debate=debate, model=judge_model
                    )
                    all_judgements.append(judgment)

            return all_judgements

        except Exception as e:
            self.logger.error(f"Judging failed: {str(e)}")
            return None

    def _find_winner(
        self, judgement_list: List[JudgeResult]
    ) -> Tuple[Literal["opposition", "proposition"], float]:
        """Calculate the winner and margin from the judge results."""
        # Calculate confidence sums for each side
        prop_confidence = sum(
            judge.confidence
            for judge in judgement_list
            if judge.winner == "proposition"
        )
        opp_confidence = sum(
            judge.confidence
            for judge in judgement_list
            if judge.winner == "opposition"
        )
        total_confidence = prop_confidence + opp_confidence

        # Determine winner by total confidence
        winner: Literal["opposition", "proposition"]
        if prop_confidence > opp_confidence:
            winner = "proposition"
            winner_confidence = prop_confidence
            loser_confidence = opp_confidence
        elif opp_confidence > prop_confidence:
            winner = "opposition"
            winner_confidence = opp_confidence
            loser_confidence = prop_confidence
        else:
            # In case of tie in confidence, random selection
            winner = random.choice(["proposition", "opposition"])
            winner_confidence = prop_confidence  # Equal to opp_confidence in this case
            loser_confidence = prop_confidence  # Equal to winner_confidence in this case
            self.logger.warning(
                "Debate resulted in a tie in confidence - randomly selecting winner"
            )

        # Calculate margin as winner confidence - loser confidence
        if total_confidence > 0:
            margin = (winner_confidence - loser_confidence) / total_confidence
        else:
            margin = 0.0

        return (winner, margin)

    def extract_bet_history(self, debate_path: Path) -> Dict[str, List[int]]:
        """Extract betting history from the debate."""
        try:
            debate = self.config.debate_service.continue_debate(debate_path)

            if not debate.debator_bets:
                self.logger.warning(f"No bets found in debate: {debate_path}")
                return {}

            prop_bets = []
            opp_bets = []

            # Sort bets by speech type to maintain chronological order
            speech_order = ["opening", "rebuttal", "closing"]

            for speech_type in speech_order:
                for bet in debate.debator_bets:
                    if bet.speech_type.value == speech_type:
                        if bet.side.value == "proposition":
                            prop_bets.append(bet.amount)
                        else:
                            opp_bets.append(bet.amount)

            return {
                debate.proposition_model: prop_bets,
                debate.opposition_model: opp_bets,
            }
        except Exception as e:
            self.logger.error(f"Failed to extract bets: {str(e)}")
            return {}

    def generate_debate_filename(self, prop_model: str, opp_model: str) -> str:
        """Generate a unique filename with timestamp for the debate."""
        prop_name = sanitize_model_name(prop_model)
        opp_name = sanitize_model_name(opp_model)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prop_name}_vs_{opp_name}_{timestamp}.json"

    def run_all_debates(self):
        """Run all debates from the balance_debate.json file."""
        self.logger.info(f"Starting to run {len(self.debate_pairings)} balanced debates")

        # Randomize topics if needed
        if len(self.debate_pairings) > len(self.topics):
            self.logger.warning(
                f"More debates ({len(self.debate_pairings)}) than available topics ({len(self.topics)}). Some topics will be reused."
            )
            topics_to_use = []
            for i in range(len(self.debate_pairings)):
                topics_to_use.append(self.topics[i % len(self.topics)])
        else:
            topics_to_use = random.sample(self.topics, k=len(self.debate_pairings))

        # Run each debate
        for i, (match, topic) in enumerate(zip(self.debate_pairings, topics_to_use)):
            self.logger.info(f"Debate {i+1}/{len(self.debate_pairings)}")

            # Generate output path
            filename = self.generate_debate_filename(match["proposition"], match["opposition"])
            output_path = self.output_dir / filename

            # Run the debate
            success = self.run_debate(match, topic, output_path)

            if not success or not checkIfComplete(output_path):
                self.logger.warning(f"Debate might be incomplete: {output_path}")
                try:
                    self.logger.info("Attempting to continue the debate...")
                    self.config.debate_service.continue_debate(output_path)
                    self.logger.info("Successfully continued and completed the debate")
                except Exception as e:
                    self.logger.error(f"Failed to continue debate: {str(e)}")
                    continue

            # Judge the debate
            judgements = self.judge_debate(output_path)

            if judgements:
                winner, margin = self._find_winner(judgements)

                # Extract bet history
                bet_history = self.extract_bet_history(output_path)

                # Log results
                self.logger.info(f"Debate result: {winner} won with margin {margin:.2f}")
                self.logger.info(f"Bet history: {bet_history}")

                # Log individual judge decisions
                self.logger.info("Judge decisions:")
                for judge in judgements:
                    self.logger.info(f"  {judge.model}: {judge.winner} (confidence: {judge.confidence})")
            else:
                self.logger.error(f"Failed to judge debate: {output_path}")

            self.logger.info(f"Completed debate {i+1}/{len(self.debate_pairings)}")

            # Optional: add a delay between debates to avoid rate limiting
            time.sleep(10)  # 10 second delay

        self.logger.info("All debates completed!")
        self.logger.info(f"Output files are saved in the {self.output_dir} directory")


def main():
    """Run all debates from the balanced schedule."""
    load_dotenv()
    config = Config()

    runner = BalancedDebateRunner(config)
    runner.run_all_debates()


if __name__ == "__main__":
    main()
