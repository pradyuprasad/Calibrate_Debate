import logging
from pathlib import Path
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from core.models import (
    DebateTotal,
    DebateTopic,
    Round,
    Side,
)
from utils.utils import make_rounds
from core.api_client import OpenRouterClient
from core.message_formatter import MessageFormatter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DebateService:
    def __init__(
        self,
        api_client: OpenRouterClient,
        message_formatter: MessageFormatter
    ):
        self.api_client = api_client
        self.message_formatter = message_formatter

    def run_debate(
        self,
        proposition_model: str,
        opposition_model: str,
        motion: DebateTopic,
        path_to_store: Path,
    ) -> DebateTotal:
        """
        Run a complete debate including all speeches and judgments
        """
        logger.debug("running debate!")
        logger.info(f"Starting debate on motion: {motion.topic_description}")

        # Initialize debate
        debate = DebateTotal(
            motion=motion,
            proposition_model=proposition_model,
            opposition_model=opposition_model,
            prompts=self.message_formatter.prompts,
            path_to_store=path_to_store,
        )

        # Run debate rounds
        rounds = make_rounds()
        for round in rounds:
            logger.info(f"Executing round: {round.speech_type} for {round.side}")
            self._execute_round(debate, round)
            debate.save_to_json()


        logger.info("Debate completed successfully")
        return debate

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=10, max=20),
        before_sleep=lambda retry_state: logger.warning(
            f"Attempt {retry_state.attempt_number} failed. Retrying..."
        ),
    )
    def _execute_round(self, debate: DebateTotal, round: Round):
        """
        Execute a single debate round, handling API communication and state updates
        """
        model = (
            debate.proposition_model
            if round.side == Side.PROPOSITION
            else debate.opposition_model
        )

        messages = self.message_formatter.get_chat_messages(debate, round)

        try:
            response = self.api_client.send_request(model, messages)

            # Error checking





            if round.side == Side.PROPOSITION:
                debate.proposition_output.speeches[round.speech_type] = response.content
            else:
                debate.opposition_output.speeches[round.speech_type] = response.content

            # Track successful token usage
            debate.debator_token_counts.add_successful_call(
                model=model,
                completion_tokens=response.completion_tokens,
                prompt_tokens=response.prompt_tokens,
                total_tokens=response.completion_tokens + response.prompt_tokens
            )

            logger.info(f"Successfully completed {round.speech_type} for {round.side}")

        except Exception as e:
            # Track failed token usage if available
            if isinstance(e, requests.exceptions.RequestException) and hasattr(e, 'response'):
                usage = e.response.json().get('usage', {}) if e.response else {}
                debate.debator_token_counts.add_failed_call(
                    model=model,
                    completion_tokens=usage.get("completion_tokens", 0),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0)
                )
            logger.error(f"Error in debate round: {str(e)}")
            raise

    def continue_debate(self, debate_path: Path) -> DebateTotal:
        """
        Continues a partially completed debate by completing any missing speeches.

        Args:
            debate_path: Path to the JSON file containing the partial debate

        Returns:
            DebateTotal: The completed debate
        """
        logger.info(f"Continuing debate from {debate_path}")

        # Load the existing debate
        debate = DebateTotal.load_from_json(debate_path)

        # Get all rounds that need to be completed
        rounds = make_rounds()
        incomplete_rounds = []

        for round in rounds:
            if round.side == Side.PROPOSITION:
                speech = debate.proposition_output.speeches[round.speech_type]
            else:
                speech = debate.opposition_output.speeches[round.speech_type]

            if speech == -1:  # Speech is missing
                incomplete_rounds.append(round)

        # Complete missing rounds
        for round in incomplete_rounds:
            logger.info(f"Completing missing {round.speech_type} for {round.side}")
            self._execute_round(debate, round)
            debate.save_to_json()

        logger.info("Debate continuation completed")
        return debate


