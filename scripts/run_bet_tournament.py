#!/usr/bin/env python3
"""
Simplified betting tournament script that runs debates with private confidence betting.
Uses predefined pairings from previous tournament results.
"""
import logging
import json
from pathlib import Path
import random
from datetime import datetime
from typing import List, Dict, Literal, Optional, Tuple

from dotenv import load_dotenv
from core.models import (
    DebateTopic,
    DebateType
)
from config import Config
from topics.load_topics import load_topics
from scripts.utils import sanitize_model_name, checkIfComplete


def setup_logging():
    """Configure logging for the tournament."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("tournament")


class BetTournament:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging()
        self.tournament_dir = config.tournament_dir / f"bet_tournament_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.tournament_dir.mkdir(exist_ok=True, parents=True)

        # Load topics
        self.topics = load_topics(config)

        # Configure judge model - use a consistent judge
        self.judge_model1 = "openai/o1-mini"  # Adjust as needed
        self.judge_model2 = "deepseek/deepseek-chat"

        # Define models based on your provided list
        self.models = [
            "deepseek/deepseek-r1",
            "google/gemini-2.0-flash-thinking-exp:free",
            "openai/chatgpt-4o-latest",
            "anthropic/claude-3.5-sonnet",
            "qwen/qwen-max",
            "deepseek/deepseek-chat",
            "openai/gpt-4o-mini",
            "openai/o1-mini",
            "google/gemini-2.0-flash-001",
            "meta-llama/llama-3.3-70b-instruct",
            "google/gemini-2.0-pro-exp-02-05:free",
            "anthropic/claude-3.5-haiku",
            "google/gemma-2-27b-it"
        ]

        # Tournament status tracking
        self.model_stats = {model: {"wins": 0, "losses": 0, "bet_history": []} for model in self.models}

    def load_predefined_matches(self) -> Dict[int, List[Dict]]:
        """Load the predefined matches from your provided data."""
        predefined_matches = {
            1: [
                {"prop_model": "openai/gpt-4o-mini", "opp_model": "anthropic/claude-3.5-haiku"},
                {"prop_model": "google/gemini-2.0-flash-001", "opp_model": "deepseek/deepseek-chat"},
                {"prop_model": "openai/o1-mini", "opp_model": "google/gemini-2.0-flash-thinking-exp:free"},
                {"prop_model": "openai/chatgpt-4o-latest", "opp_model": "google/gemini-2.0-pro-exp-02-05:free"},
                {"prop_model": "google/gemma-2-27b-it", "opp_model": "deepseek/deepseek-r1"},
                {"prop_model": "meta-llama/llama-3.3-70b-instruct", "opp_model": "qwen/qwen-max"}
            ],
            2: [
                {"prop_model": "deepseek/deepseek-r1", "opp_model": "anthropic/claude-3.5-haiku"},
                {"prop_model": "openai/chatgpt-4o-latest", "opp_model": "qwen/qwen-max"},
                {"prop_model": "google/gemini-2.0-flash-001", "opp_model": "google/gemini-2.0-pro-exp-02-05:free"},
                {"prop_model": "google/gemma-2-27b-it", "opp_model": "openai/o1-mini"},
                {"prop_model": "meta-llama/llama-3.3-70b-instruct", "opp_model": "openai/gpt-4o-mini"},
                {"prop_model": "deepseek/deepseek-chat", "opp_model": "anthropic/claude-3.5-sonnet"}
            ],
            3: [
                {"prop_model": "openai/gpt-4o-mini", "opp_model": "qwen/qwen-max"},
                {"prop_model": "google/gemini-2.0-flash-001", "opp_model": "openai/o1-mini"},
                {"prop_model": "anthropic/claude-3.5-haiku", "opp_model": "deepseek/deepseek-chat"},
                {"prop_model": "google/gemini-2.0-pro-exp-02-05:free", "opp_model": "google/gemini-2.0-flash-thinking-exp:free"},
                {"prop_model": "google/gemma-2-27b-it", "opp_model": "anthropic/claude-3.5-sonnet"},
                {"prop_model": "deepseek/deepseek-r1", "opp_model": "openai/chatgpt-4o-latest"}
            ]
        }
        return predefined_matches

    def run_debate(self, match: Dict, topic: DebateTopic, output_path: Path) -> bool:
        """Run a debate between two models with private betting."""
        self.logger.info(f"Starting debate: {match['prop_model']} vs {match['opp_model']}")
        self.logger.info(f"Topic: {topic.topic_description}")

        try:
            self.config.debate_service.run_debate(
                proposition_model=match['prop_model'],
                opposition_model=match['opp_model'],
                motion=topic,
                path_to_store=output_path,
                debate_type=DebateType.PRIVATE_BET
            )
            return True
        except Exception as e:
            self.logger.error(f"Debate failed: {str(e)}")
            return False

    def judge_debate(self, debate_path: Path) -> Optional[List[Literal['opposition', 'proposition']]]:
        """Judge a debate and return the winner."""
        self.logger.info(f"Judging debate: {debate_path}")

        try:
            winners: List[Literal['opposition', 'proposition']] = []
            debate = self.config.debate_service.continue_debate(debate_path)

            for i in range(3):

                judgment = self.config.judgement_processor.process_judgment(
                    debate=debate,
                    model=self.judge_model1
                )
                judgment = self.config.judgement_processor.process_judgment(
                    debate=debate,
                    model=self.judge_model2
                )
                winners.append(judgment.winner)

            return winners
        except Exception as e:
            self.logger.error(f"Judging failed: {str(e)}")
            return None

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
                debate.opposition_model: opp_bets
            }
        except Exception as e:
            self.logger.error(f"Failed to extract bets: {str(e)}")
            return {}

    def save_round_results(self, round_num: int, results: List[Dict]):
        """Save round results to a JSON file."""
        results_path = self.tournament_dir / f"round_{round_num}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Saved round {round_num} results to {results_path}")

    def _find_winner(self, judgement_list: List[Literal['opposition', 'proposition']]) -> Tuple[Literal['opposition', 'proposition'], float]:
        prop_count = judgement_list.count('proposition')
        opp_count = judgement_list.count('opposition')

        winner : Literal['opposition', 'proposition']
        if prop_count > opp_count:
            winner = 'proposition'
        elif opp_count > prop_count:
            winner = 'opposition'
        else:
            raise ValueError("Tied!")

        winner_margin = judgement_list.count(winner)

        margin_normalised = winner_margin / len(judgement_list)

        return (winner, margin_normalised)


    def run_tournament(self):
        """Run the tournament using predefined pairings."""
        self.logger.info("Starting betting tournament")

        predefined_matches = self.load_predefined_matches()

        for round_num in range(1, 4):  # 3 rounds
            self.logger.info(f"=== ROUND {round_num} ===")

            # Create round directory
            round_dir = self.tournament_dir / f"round_{round_num}"
            round_dir.mkdir(exist_ok=True)

            # Get matches for this round
            round_matches = predefined_matches[round_num]

            # Select topics
            topics = random.sample(self.topics, k=len(round_matches))

            # Run debates
            round_results = []

            for i, (match, topic) in enumerate(zip(round_matches, topics)):
                self.logger.info(f"Match {i+1}/{len(round_matches)}: {match['prop_model']} vs {match['opp_model']}")

                prop_name = sanitize_model_name(match['prop_model'])
                opp_name = sanitize_model_name(match['opp_model'])
                debate_path = round_dir / f"{prop_name}_vs_{opp_name}.json"

                # Run the debate
                success = self.run_debate(match, topic, debate_path)

                if not success or not checkIfComplete(debate_path):
                    self.logger.warning("Debate incomplete, attempting to continue")
                    try:
                        self.config.debate_service.continue_debate(debate_path)
                    except Exception as e:
                        self.logger.error(f"Failed to continue debate: {str(e)}")
                        continue

                # Judge the debate
                winners = self.judge_debate(debate_path)

                winner, margin = self._find_winner(winners)


                if not winner:
                    self.logger.error(f"Failed to get winner for debate: {debate_path}")
                    continue

                # Update stats
                if winners == "proposition":
                    self.model_stats[match['prop_model']]["wins"] += 1
                    self.model_stats[match['opp_model']]["losses"] += 1
                else:
                    self.model_stats[match['opp_model']]["wins"] += 1
                    self.model_stats[match['prop_model']]["losses"] += 1

                # Extract bet history
                bet_history = self.extract_bet_history(debate_path)

                for model, bets in bet_history.items():
                    if model in self.model_stats and bets:
                        self.model_stats[model]["bet_history"].append(bets)

                # Record result
                result = {
                    "match_id": i + 1,
                    "proposition": match['prop_model'],
                    "opposition": match['opp_model'],
                    "topic": topic.topic_description,
                    "winner": winner,
                    "debate_path": str(debate_path),
                    "bet_history": bet_history,
                    "margin": margin
                }

                round_results.append(result)
                self.logger.info(f"Match result: {winner} won with bet history: {bet_history}")

            # Save round results
            self.save_round_results(round_num, round_results)

            # Print current standings
            self.logger.info("=== CURRENT STANDINGS ===")
            sorted_models = sorted(
                self.model_stats.items(),
                key=lambda x: (x[1]["wins"] - x[1]["losses"], x[1]["wins"]),
                reverse=True
            )

            for model, stats in sorted_models:
                bet_trends = []
                for bet_series in stats.get("bet_history", []):
                    if bet_series:
                        # Show confidence evolution (first bet -> last bet)
                        bet_trends.append(f"{bet_series[0]}->{bet_series[-1]}")

                bet_trend_str = ", ".join(bet_trends) if bet_trends else "No bets"

                self.logger.info(
                    f"{model}: {stats['wins']}W-{stats['losses']}L | Bet trends: {bet_trend_str}"
                )

        # Save final tournament results
        final_path = self.tournament_dir / "tournament_results.json"
        with open(final_path, "w") as f:
            json.dump({
                "model_stats": self.model_stats,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

        self.logger.info(f"Tournament complete! Results saved to {final_path}")
        self.print_confidence_evolution()

    def print_confidence_evolution(self):
        """Print analysis of confidence evolution during debates."""
        self.logger.info("=== CONFIDENCE EVOLUTION ANALYSIS ===")

        for model, stats in self.model_stats.items():
            bet_histories = stats.get("bet_history", [])
            if not bet_histories:
                continue

            total_debates = len(bet_histories)
            increased_confidence = 0
            decreased_confidence = 0
            unchanged_confidence = 0

            for bets in bet_histories:
                if len(bets) >= 2:
                    if bets[-1] > bets[0]:
                        increased_confidence += 1
                    elif bets[-1] < bets[0]:
                        decreased_confidence += 1
                    else:
                        unchanged_confidence += 1

            self.logger.info(f"\n{model} confidence patterns ({total_debates} debates):")
            if total_debates > 0:
                self.logger.info(f"  Increased confidence: {increased_confidence} ({increased_confidence/total_debates*100:.1f}%)")
                self.logger.info(f"  Decreased confidence: {decreased_confidence} ({decreased_confidence/total_debates*100:.1f}%)")
                self.logger.info(f"  Unchanged confidence: {unchanged_confidence} ({unchanged_confidence/total_debates*100:.1f}%)")

                # Show all bet sequences
                self.logger.info("  Bet sequences:")
                for i, bets in enumerate(bet_histories):
                    self.logger.info(f"    Debate {i+1}: {bets}")


def main():
    """Run the tournament with private betting."""
    load_dotenv()
    config = Config()

    tournament = BetTournament(config)
    tournament.run_tournament()


if __name__ == "__main__":
    main()
