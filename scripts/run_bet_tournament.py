#!/usr/bin/env python3
"""
Betting tournament script that runs debates with private confidence betting.
Only uses predefined pairings for round 1, then dynamically pairs models based on performance.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

from dotenv import load_dotenv

from ai_models.load_models import load_debate_models
from config import Config
from core.models import DebateTopic, DebateType, JudgeResult
from scripts.utils import checkIfComplete, sanitize_model_name
from topics.load_topics import load_topics


class ModelStats(TypedDict):
    wins: int
    losses: int
    bet_history: List[List[int]]
    total_margin: float
    win_margin: float
    rounds_played: int
    opponents: List[str]


def setup_logging(config: Config):
   """Configure logging for the tournament to output to both a file and stdout."""
   logger = config.logger.get_logger()
   return logger


class BetTournament:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.tournament_dir = (
            config.tournament_dir
            / f"bet_tournament_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        self.tournament_dir.mkdir(exist_ok=True, parents=True)

        # Load topics
        self.topics = load_topics(config)

        # Configure judge models as a dynamic list
        self.judge_models = [
            "qwen/qwq-32b",
            "google/gemini-pro-1.5",
            "deepseek/deepseek-chat",
        ]

        # Log which judges will be used
        self.logger.info(
            f"Using {len(self.judge_models)} judges for evaluation: {', '.join(self.judge_models)}"
        )

        # Load models from configuration
        self.models_data = load_debate_models(config)
        self.models = list(self.models_data.keys())

        self.num_rounds = 5

        self.voting_rounds = 2

        self.model_stats: Dict[str, ModelStats] = {
            model: {
                "wins": 0,
                "losses": 0,
                "bet_history": [],
                "total_margin": 0.0,
                "win_margin": 0.0,
                "rounds_played": 0,
                "opponents": [],
            }
            for model in self.models
        }

    def load_predefined_round1_matches(self) -> List[Dict]:
        """Load only the predefined matches for round 1 from a configuration file."""
        try:
            # Path to the round 1 matches file (adjust path as needed)
            matches_file_path = getattr(self.config, "round1_matches_path", None)

            if matches_file_path and Path(matches_file_path).exists():
                with open(matches_file_path) as f:
                    round1_matches = json.load(f)

                self.logger.info(
                    f"Successfully loaded {len(round1_matches)} matches for round 1"
                )
                return round1_matches
            else:
                return self._generate_random_round1_pairings()
        except Exception as e:
            self.logger.error(f"Failed to load predefined matches: {str(e)}")
            return self._generate_random_round1_pairings()

    def _generate_random_round1_pairings(self) -> List[Dict]:
        """Generate random pairings for round 1 if no predefined matches exist."""
        models_list = list(self.models)
        random.shuffle(models_list)
        matches = []

        # Create pairs
        for i in range(0, len(models_list), 2):
            if i + 1 < len(models_list):
                matches.append(
                    {"prop_model": models_list[i], "opp_model": models_list[i + 1]}
                )

        self.logger.info(f"Generated {len(matches)} random matches for round 1")
        return matches

    def generate_subsequent_round_pairings(self, round_num: int) -> List[Dict]:
        """Generate pairings for rounds after round 1 based on current standings."""
        self.logger.info(
            f"Generating pairings for round {round_num} based on current standings"
        )

        # Sort models by performance (win-loss record, then total margin)
        sorted_models = sorted(
            self.models,
            key=lambda model: (
                self.model_stats[model]["wins"] - self.model_stats[model]["losses"],
                self.model_stats[model]["total_margin"],
            ),
            reverse=True,
        )

        # Group models by similar performance
        # We'll create groups of similar-performing models
        groups = []
        current_group = []

        for i, model in enumerate(sorted_models):
            if i == 0 or (
                self.model_stats[model]["wins"] - self.model_stats[model]["losses"]
                == self.model_stats[sorted_models[i - 1]]["wins"]
                - self.model_stats[sorted_models[i - 1]]["losses"]
            ):
                current_group.append(model)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [model]

        if current_group:
            groups.append(current_group)

        # Shuffle each group to add variety
        for group in groups:
            random.shuffle(group)

        # Create pairings within groups when possible
        matches = []
        unpaired = []

        for group in groups:
            # If odd number in group, save one for inter-group pairing
            if len(group) % 2 == 1:
                unpaired.append(group.pop())

            # Pair models within the group
            for i in range(0, len(group), 2):
                if i + 1 < len(group):
                    # Check if these models have faced each other already
                    if group[i + 1] in self.model_stats[group[i]]["opponents"]:
                        # If they've already faced each other, add both to unpaired
                        unpaired.extend([group[i], group[i + 1]])
                    else:
                        matches.append(
                            {"prop_model": group[i], "opp_model": group[i + 1]}
                        )

        # Handle unpaired models
        random.shuffle(unpaired)
        for i in range(0, len(unpaired), 2):
            if i + 1 < len(unpaired):
                # Check if these models have faced each other already
                if unpaired[i + 1] in self.model_stats[unpaired[i]]["opponents"]:
                    # If we're at the last pair and they've faced each other,
                    # we'll have to allow a rematch
                    if i + 2 >= len(unpaired):
                        matches.append(
                            {"prop_model": unpaired[i], "opp_model": unpaired[i + 1]}
                        )
                else:
                    matches.append(
                        {"prop_model": unpaired[i], "opp_model": unpaired[i + 1]}
                    )

        self.logger.info(f"Generated {len(matches)} matches for round {round_num}")
        return matches

    def run_debate(self, match: Dict, topic: DebateTopic, output_path: Path) -> bool:
        """Run a debate between two models with private betting."""
        self.logger.info(
            f"Starting debate: {match['prop_model']} vs {match['opp_model']}"
        )
        self.logger.info(f"Topic: {topic.topic_description}")

        try:
            self.config.debate_service.run_debate(
                proposition_model=match["prop_model"],
                opposition_model=match["opp_model"],
                motion=topic,
                path_to_store=output_path,
                debate_type=DebateType.PRIVATE_BET,
            )
            return True
        except Exception as e:
            self.logger.error(f"Debate failed: {str(e)}")
            return False

    def judge_debate(self, debate_path: Path) -> Optional[List[JudgeResult]]:
        """Judge a debate and return the winner using all available judge models."""
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

    def save_round_results(self, round_num: int, results: List[Dict]):
        """Save round results to a JSON file."""
        results_path = self.tournament_dir / f"round_{round_num}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Saved round {round_num} results to {results_path}")

    def _find_winner(
        self, judgement_list: List[JudgeResult]
    ) -> Tuple[Literal["opposition", "proposition"], float]:
        # Calculate confidence sums for each side
        prop_confidence = sum(
            judge.confidence
            for judge in judgement_list
            if judge.winner == "proposition"
        )
        opp_confidence = sum(
            judge.confidence for judge in judgement_list if judge.winner == "opposition"
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
            loser_confidence = (
                prop_confidence  # Equal to winner_confidence in this case
            )
            self.logger.warning(
                "Debate resulted in a tie in confidence - randomly selecting winner"
            )

        # Calculate margin as winner confidence - loser confidence
        if total_confidence > 0:
            margin = (winner_confidence - loser_confidence) / total_confidence
        else:
            margin = 0.0

        return (winner, margin)

    def run_tournament(self):
        """Run the tournament with dynamic pairings after round 1."""
        self.logger.info(f"Starting betting tournament with {self.num_rounds} rounds")

        for round_num in range(1, self.num_rounds + 1):
            self.logger.info(f"=== ROUND {round_num} ===")

            # Create round directory
            round_dir = self.tournament_dir / f"round_{round_num}"
            round_dir.mkdir(exist_ok=True)

            # Get matches for this round
            if round_num == 1:
                round_matches = self.load_predefined_round1_matches()
            else:
                round_matches = self.generate_subsequent_round_pairings(round_num)

            # Select topics
            if len(round_matches) > len(self.topics):
                self.logger.warning(
                    f"More matches ({len(round_matches)}) than available topics ({len(self.topics)}). Some topics will be reused."
                )
                random.shuffle(self.topics)
                topics = []
                for i in range(len(round_matches)):
                    topics.append(self.topics[i % len(self.topics)])
            else:
                topics = random.sample(self.topics, k=len(round_matches))

            # Run debates
            round_results = []

            for i, (match, topic) in enumerate(zip(round_matches, topics)):
                self.logger.info(
                    f"Match {i + 1}/{len(round_matches)}: {match['prop_model']} vs {match['opp_model']}"
                )

                # Update tracking of opponents
                self.model_stats[match["prop_model"]]["opponents"].append(
                    match["opp_model"]
                )
                self.model_stats[match["opp_model"]]["opponents"].append(
                    match["prop_model"]
                )

                prop_name = sanitize_model_name(match["prop_model"])
                opp_name = sanitize_model_name(match["opp_model"])
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
                judgements = self.judge_debate(debate_path)

                if not judgements:
                    self.logger.error(
                        f"Failed to get judgements for debate: {debate_path}"
                    )
                    continue

                winner, margin = self._find_winner(judgements)

                # Update stats
                prop_model = match["prop_model"]
                opp_model = match["opp_model"]

                # Update rounds played
                self.model_stats[prop_model]["rounds_played"] += 1
                self.model_stats[opp_model]["rounds_played"] += 1

                if winner == "proposition":
                    self.model_stats[prop_model]["wins"] += 1
                    self.model_stats[prop_model]["total_margin"] += margin
                    self.model_stats[prop_model]["win_margin"] += margin

                    self.model_stats[opp_model]["losses"] += 1
                    self.model_stats[opp_model]["total_margin"] -= margin
                else:
                    self.model_stats[opp_model]["wins"] += 1
                    self.model_stats[opp_model]["total_margin"] += margin
                    self.model_stats[opp_model]["win_margin"] += margin

                    self.model_stats[prop_model]["losses"] += 1
                    self.model_stats[prop_model]["total_margin"] -= margin

                # Extract bet history
                bet_history = self.extract_bet_history(debate_path)

                for model, bets in bet_history.items():
                    if model in self.model_stats and bets:
                        self.model_stats[model]["bet_history"].append(bets)

                # Record result
                result = {
                    "match_id": i + 1,
                    "proposition": match["prop_model"],
                    "opposition": match["opp_model"],
                    "topic": topic.topic_description,
                    "winner": winner,
                    "margin": margin,
                    "judgements": [judgement.to_dict() for judgement in judgements],
                    "judge_models": self.judge_models,
                    "debate_path": str(debate_path),
                    "bet_history": bet_history,
                }

                round_results.append(result)
                self.logger.info(
                    f"Match result: {winner} won with margin {margin:.2f} and bet history: {bet_history}"
                )

            # Save round results
            self.save_round_results(round_num, round_results)

            # Print current standings
            self.logger.info("=== CURRENT STANDINGS ===")
            sorted_models = sorted(
                self.model_stats.items(),
                key=lambda x: (x[1]["wins"] - x[1]["losses"], x[1]["total_margin"]),
                reverse=True,
            )

            for model, stats in sorted_models:
                bet_trends = []
                for bet_series in stats.get("bet_history", []):
                    if bet_series:
                        # Show confidence evolution (first bet -> last bet)
                        bet_trends.append(f"{bet_series[0]}->{bet_series[-1]}")

                bet_trend_str = ", ".join(bet_trends) if bet_trends else "No bets"

                self.logger.info(
                    f"{model}: {stats['wins']}W-{stats['losses']}L (margin: {stats['total_margin']:.2f}) | Bet trends: {bet_trend_str}"
                )

        # Save final tournament results
        final_path = self.tournament_dir / "tournament_results.json"
        with open(final_path, "w") as f:
            json.dump(
                {
                    "model_stats": self.model_stats,
                    "judge_models": self.judge_models,
                    "voting_rounds": self.voting_rounds,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        self.logger.info(f"Tournament complete! Results saved to {final_path}")
        self.print_confidence_evolution()
        self.print_final_rankings()

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

            self.logger.info(
                f"\n{model} confidence patterns ({total_debates} debates):"
            )
            if total_debates > 0:
                self.logger.info(
                    f"  Increased confidence: {increased_confidence} ({increased_confidence / total_debates * 100:.1f}%)"
                )
                self.logger.info(
                    f"  Decreased confidence: {decreased_confidence} ({decreased_confidence / total_debates * 100:.1f}%)"
                )
                self.logger.info(
                    f"  Unchanged confidence: {unchanged_confidence} ({unchanged_confidence / total_debates * 100:.1f}%)"
                )

                # Show all bet sequences
                self.logger.info("  Bet sequences:")
                for i, bets in enumerate(bet_histories):
                    self.logger.info(f"    Debate {i + 1}: {bets}")

    def print_final_rankings(self):
        """Print final tournament rankings."""
        self.logger.info("=== FINAL TOURNAMENT RANKINGS ===")

        # Sort by win-loss record first, then by margin
        sorted_models = sorted(
            self.model_stats.items(),
            key=lambda x: (
                x[1]["wins"] - x[1]["losses"],
                x[1]["total_margin"],
                x[1]["win_margin"],
            ),
            reverse=True,
        )

        for rank, (model, stats) in enumerate(sorted_models, 1):
            win_rate = (
                stats["wins"] / stats["rounds_played"] * 100
                if stats["rounds_played"] > 0
                else 0
            )

            self.logger.info(
                f"{rank}. {model}: {stats['wins']}W-{stats['losses']}L (Win rate: {win_rate:.1f}%, Margin: {stats['total_margin']:.2f})"
            )


def main():
    """Run the tournament with private betting."""
    load_dotenv()
    config = Config()

    tournament = BetTournament(config)
    tournament.run_tournament()


if __name__ == "__main__":
    main()
