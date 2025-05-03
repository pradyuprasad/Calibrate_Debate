#!/usr/bin/env python3
"""
Script to run a tournament where each model debates against itself with private betting.
Each model will debate itself 3 times, cycling through available topics.
"""

import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from ai_models.load_models import load_debate_models
from config import Config
from core.models import DebateType
from scripts.utils import sanitize_model_name
from topics.load_topics import load_topics


def setup_logging(config: Config):
    """Configure logging for the tournament."""
    logger = config.logger.get_logger()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = f"private_self_tournament_{timestamp}.log"
    config.logger.set_log_file(log_file)
    return logger


class PrivateSelfTournamentRunner:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        
        # Create output directory for private self debates
        self.output_dir = Path("private_self_debates")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load topics and models
        self.topics = load_topics(config)
        self.models = list(load_debate_models(config).keys())
        self.logger.info(f"Loaded {len(self.models)} models and {len(self.topics)} topics")
        
        # Tournament settings
        self.debates_per_model = 3
        
        # Results tracking
        self.results = {
            model: {
                "debates": [],
                "prop_bets": [],
                "opp_bets": []
            } for model in self.models
        }

    def run_debate(self, model: str, topic_idx: int) -> bool:
        """Run a single self-debate with private betting."""
        topic = self.topics[topic_idx % len(self.topics)]
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_model_name = sanitize_model_name(model)
        filename = f"{clean_model_name}_vs_self_{timestamp}.json"
        output_path = self.output_dir / filename
        
        self.logger.info(f"Starting self-debate for {model}")
        self.logger.info(f"Topic: {topic.topic_description}")
        
        try:
            debate = self.config.debate_service.run_debate(
                proposition_model=model,
                opposition_model=model,
                motion=topic,
                path_to_store=output_path,
                debate_type=DebateType.PRIVATE_SAME_DEBATOR
            )
            
            # Track betting behavior
            if debate.debator_bets:
                for bet in debate.debator_bets:
                    if bet.side.value == "proposition":
                        self.results[model]["prop_bets"].append(bet.amount)
                    else:
                        self.results[model]["opp_bets"].append(bet.amount)
            
            # Store debate results
            debate_result = {
                "topic": topic.topic_description,
                "path": str(output_path),
                "bet_history": {
                    "proposition": [bet.amount for bet in debate.debator_bets if bet.side.value == "proposition"],
                    "opposition": [bet.amount for bet in debate.debator_bets if bet.side.value == "opposition"]
                } if debate.debator_bets else {}
            }
            self.results[model]["debates"].append(debate_result)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Debate failed: {str(e)}")
            return False

    def run_tournament(self):
        """Run the complete tournament."""
        self.logger.info("Starting private self-debate tournament")
        
        for model in self.models:
            self.logger.info(f"\n=== Running debates for {model} ===")
            
            for i in range(self.debates_per_model):
                topic_idx = i
                success = self.run_debate(model, topic_idx)
                
                if not success:
                    self.logger.error(f"Failed to complete debate {i+1} for {model}")
                    continue
                
                self.logger.info(f"Completed debate {i+1}/{self.debates_per_model} for {model}")
        
        # Save tournament results
        self.save_results()
        self.print_summary()

    def save_results(self):
        """Save tournament results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        results_path = self.output_dir / f"tournament_summary_{timestamp}.json"
        
        with open(results_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "models": self.models,
                "debates_per_model": self.debates_per_model,
                "results": self.results
            }, f, indent=2)
        
        self.logger.info(f"Saved tournament results to {results_path}")

    def print_summary(self):
        """Print a summary of the tournament results."""
        self.logger.info("\n=== Tournament Summary ===")
        
        for model in self.models:
            stats = self.results[model]
            total_debates = len(stats["debates"])
            
            self.logger.info(f"\nModel: {model}")
            self.logger.info(f"Total debates: {total_debates}")
            
            if stats["prop_bets"]:
                avg_prop_bet = sum(stats["prop_bets"]) / len(stats["prop_bets"])
                avg_opp_bet = sum(stats["opp_bets"]) / len(stats["opp_bets"])
                self.logger.info(f"Average proposition bet: {avg_prop_bet:.1f}")
                self.logger.info(f"Average opposition bet: {avg_opp_bet:.1f}")
                
                # Show bet evolution
                self.logger.info("\nBet sequences:")
                for i, debate in enumerate(stats["debates"]):
                    if "bet_history" in debate:
                        prop_bets = debate["bet_history"].get("proposition", [])
                        opp_bets = debate["bet_history"].get("opposition", [])
                        self.logger.info(f"Debate {i+1}:")
                        self.logger.info(f"  Proposition bets: {prop_bets}")
                        self.logger.info(f"  Opposition bets: {opp_bets}")


def main():
    """Run the private self-debate tournament."""
    load_dotenv()
    config = Config()
    
    tournament = PrivateSelfTournamentRunner(config)
    tournament.run_tournament()


if __name__ == "__main__":
    main()