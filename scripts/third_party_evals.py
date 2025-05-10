from dotenv import load_dotenv
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.api_client import OpenRouterClient
from core.models import DebateTotal, Side, SpeechType, APIResponse, DebatorBet
from core.logger import LoggerFactory

# Initialize logger
logger = LoggerFactory.get_logger("debate_analysis")

load_dotenv()

# Judge criteria text to append
JUDGE_CRITERIA = """
The judge will evaluate your speech using these strict criteria:
DIRECT CLASH ANALYSIS
- Every disagreement must be explicitly quoted and directly addressed
- Simply making new arguments without engaging opponents' points will be penalized
- Show exactly how your evidence/reasoning defeats theirs
- Track and reference how arguments evolve through the debate
EVIDENCE QUALITY HIERARCHY
1. Strongest: Specific statistics, named examples, verifiable cases with dates/numbers
2. Medium: Expert testimony with clear sourcing
3. Weak: General examples, unnamed cases, theoretical claims without support
- Correlation vs. causation will be scrutinized - prove causal links
- Evidence must directly support the specific claim being made
LOGICAL VALIDITY
- Each argument requires explicit warrants (reasons why it's true)
- All logical steps must be clearly shown, not assumed
- Internal contradictions severely damage your case
- Hidden assumptions will be questioned if not defended
RESPONSE OBLIGATIONS
- Every major opposing argument must be addressed
- Dropped arguments are considered conceded
- Late responses (in final speech) to early arguments are discounted
- Shifting or contradicting your own arguments damages credibility
IMPACT ANALYSIS & WEIGHING
- Explain why your arguments matter more than opponents'
- Compare competing impacts explicitly
- Show both philosophical principles and practical consequences
- Demonstrate how winning key points proves the overall motion
The judge will ignore speaking style, rhetoric, and presentation. Focus entirely on argument substance, evidence quality, and logical reasoning. Your case will be evaluated based on what you explicitly prove, not what you assume or imply.
"""

class DebateAnalyzer:
    def __init__(self, tournament_folder: str, api_key: str):
        self.tournament_folder = Path(tournament_folder)
        self.api_client = OpenRouterClient(api_key=api_key, logger=logger)
        self.results = []  # List of debate entries

    def get_all_debate_files(self) -> List[Path]:
        """Find all debate JSON files in the tournament folder"""
        return list(self.tournament_folder.glob("**/*.json"))

    def extract_debate_until_stage(self, debate: DebateTotal, stage: SpeechType, for_side: Side) -> str:
        """
        Extract debate content up to a specific stage.
        For previous stages, include speeches from both sides.
        For the current stage, only include speeches up to the side we're analyzing.
        """
        content = ""

        # Get all speech types in order
        all_speech_types = [SpeechType.OPENING, SpeechType.REBUTTAL, SpeechType.CLOSING]

        # For each speech type up to the current stage
        for speech_type in all_speech_types:
            # If we've gone past our target stage, break
            if all_speech_types.index(speech_type) > all_speech_types.index(stage):
                break

            # For previous stages, include both sides' speeches
            if speech_type != stage:
                # Add proposition speech if it exists
                prop_speech = debate.proposition_output.speeches.get(speech_type, -1)
                if prop_speech != -1:
                    content += f"\nPROPOSITION {speech_type.value.upper()}\n{prop_speech}\n\n"

                # Add opposition speech if it exists
                opp_speech = debate.opposition_output.speeches.get(speech_type, -1)
                if opp_speech != -1:
                    content += f"\nOPPOSITION {speech_type.value.upper()}\n{opp_speech}\n\n"

            # For the current stage:
            else:
                # If analyzing as proposition, show proposition speech first (if available)
                if for_side == Side.PROPOSITION:
                    prop_speech = debate.proposition_output.speeches.get(speech_type, -1)
                    if prop_speech != -1:
                        content += f"\nPROPOSITION {speech_type.value.upper()}\n{prop_speech}\n\n"

                # If analyzing as opposition, show proposition speech first, then opposition speech
                elif for_side == Side.OPPOSITION:
                    # Show proposition speech first
                    prop_speech = debate.proposition_output.speeches.get(speech_type, -1)
                    if prop_speech != -1:
                        content += f"\nPROPOSITION {speech_type.value.upper()}\n{prop_speech}\n\n"

                    # Then opposition speech
                    opp_speech = debate.opposition_output.speeches.get(speech_type, -1)
                    if opp_speech != -1:
                        content += f"\nOPPOSITION {speech_type.value.upper()}\n{opp_speech}\n\n"

        return content

    def get_original_bet(self, debate: DebateTotal, side: Side, speech_type: SpeechType) -> Optional[int]:
        """Get the original bet made by the model during the debate"""
        if not debate.debator_bets:
            return None

        for bet in debate.debator_bets:
            if bet.side == side and bet.speech_type == speech_type:
                return bet.amount

        return None

    def create_prompt(self, debate: DebateTotal, debate_content: str, as_side: Side) -> str:
        """Create a prompt for a model with debate content and judging criteria"""
        motion = debate.motion.topic_description

        prompt = f"DEBATE MOTION: {motion}\n\n"
        prompt += "=== DEBATE HISTORY ===\n"
        prompt += debate_content
        prompt += "\n\n=== JUDGING CRITERIA ===\n"
        prompt += JUDGE_CRITERIA
        prompt += f"\n\nYou are on the {as_side.value} side of this debate. Based on the debate so far, "
        prompt += "how likely do you think you are to win this debate? Provide a bet from 0-100 indicating your confidence."
        prompt += "\n\nPlease structure your response as follows:"
        prompt += "\n1. First, provide your analysis of the debate so far and why you think you would or wouldn't win."
        prompt += "\n2. Then provide your bet in the format: <bet>NUMBER</bet>"

        return prompt

    def get_model_prediction(self, model_name: str, prompt: str) -> Optional[int]:
        """Query the model and extract its prediction bet"""
        try:
            logger.info(f"Querying {model_name} for prediction")

            messages = [
                {"role": "user", "content": prompt}
            ]

            response = self.api_client.send_request(model=model_name, messages=messages)

            # Extract bet from response
            bet_match = re.search(r"<bet>(\d+)</bet>", response.content)
            if bet_match:
                bet = int(bet_match.group(1))
                # Ensure bet is within valid range
                bet = max(0, min(bet, 100))
                logger.info(f"Extracted bet: {bet}")
                return bet
            else:
                logger.warning(f"Could not extract bet from {model_name}'s response")
                logger.info(f"Response content: {response.content[:500]}... (truncated)")

                # Ask for manual input if extraction fails
                while True:
                    try:
                        user_input = input(f"\nEnter bet (0-100) for {model_name}: ")
                        bet = int(user_input.strip())
                        if 0 <= bet <= 100:
                            return bet
                        logger.error("Bet must be between 0 and 100")
                    except ValueError:
                        logger.error("Please enter a valid number")

        except Exception as e:
            logger.error(f"Error querying model {model_name}: {str(e)}")
            return None

    def analyze_debate(self, debate_file: Path):
        """Analyze a single debate file at all stages"""
        logger.info(f"Analyzing debate: {debate_file}")
        debate = DebateTotal.load_from_json(debate_file)

        prop_model = debate.proposition_model
        opp_model = debate.opposition_model

        # Create a new debate entry
        debate_entry = {
            "filename": str(debate_file.name),
            "proposition_model": prop_model,
            "opposition_model": opp_model,
            "motion": debate.motion.topic_description,
            "opening": {
                "third_party_as_prop": None,
                "third_party_as_opp": None,
                "original_as_prop": None,
                "original_as_opp": None
            },
            "rebuttal": {
                "third_party_as_prop": None,
                "third_party_as_opp": None,
                "original_as_prop": None,
                "original_as_opp": None
            },
            "closing": {
                "third_party_as_prop": None,
                "third_party_as_opp": None,
                "original_as_prop": None,
                "original_as_opp": None
            }
        }

        for stage, stage_name in [
            (SpeechType.OPENING, "opening"),
            (SpeechType.REBUTTAL, "rebuttal"),
            (SpeechType.CLOSING, "closing")
        ]:
            logger.info(f"Analyzing {stage_name} stage")

            # Skip if stage doesn't exist in debate
            if (debate.proposition_output.speeches.get(stage, -1) == -1 or
                debate.opposition_output.speeches.get(stage, -1) == -1):
                logger.info(f"Skipping {stage_name} stage - incomplete debate")
                continue

            # Get original bets from the debate
            original_prop_bet = self.get_original_bet(debate, Side.PROPOSITION, stage)
            original_opp_bet = self.get_original_bet(debate, Side.OPPOSITION, stage)

            # Store original bets
            debate_entry[stage_name]["original_as_prop"] = original_prop_bet
            debate_entry[stage_name]["original_as_opp"] = original_opp_bet

            # Get proposition model's bet when it's in proposition role
            prop_content = self.extract_debate_until_stage(debate, stage, Side.PROPOSITION)
            prop_prompt = self.create_prompt(debate, prop_content, Side.PROPOSITION)
            prop_bet = self.get_model_prediction(prop_model, prop_prompt)
            debate_entry[stage_name]["third_party_as_prop"] = prop_bet

            # Get opposition model's bet when it's in opposition role
            opp_content = self.extract_debate_until_stage(debate, stage, Side.OPPOSITION)
            opp_prompt = self.create_prompt(debate, opp_content, Side.OPPOSITION)
            opp_bet = self.get_model_prediction(opp_model, opp_prompt)
            debate_entry[stage_name]["third_party_as_opp"] = opp_bet

            # Save results after each stage
            self.save_results()
            print("saved result")

        # Add the completed debate entry to results
        self.results.append(debate_entry)

    def analyze_all_debates(self):
        """Analyze all debate files in the tournament folder"""
        debate_files = self.get_all_debate_files()
        logger.info(f"Found {len(debate_files)} debate files")

        for debate_file in debate_files:
            self.analyze_debate(debate_file)

        logger.info("Analysis complete")

    def save_results(self, output_file: str = "debate_analysis_results.json"):
        """Save analysis results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

def main():
    # Get API key from environment or prompt
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenRouter API key: ")

    # Get tournament folder
    tournament_folder = input("Enter path to private_bet_tournament folder: ")

    analyzer = DebateAnalyzer(tournament_folder, api_key)
    analyzer.analyze_all_debates()

    print(f"Analysis complete. Results saved to debate_analysis_results.json")

if __name__ == "__main__":
    main()
