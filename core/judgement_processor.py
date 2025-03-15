import logging
import sys
import re
from typing import Dict, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential

from core.models import DebateTotal, DebatePrompts, JudgeResult
from core.api_client import OpenRouterClient


class JudgementProcessor:
    """Handles processing of debate judgements"""

    def __init__(self, prompts: DebatePrompts, client: OpenRouterClient):
        self.prompts = prompts
        self.client = client
        self.logger = logging.getLogger(__name__)

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=10, max=20),
        before_sleep=lambda retry_state: logging.getLogger().warning(
            f"Attempt {retry_state.attempt_number} failed. Failed with error: {retry_state.outcome.exception() if retry_state.outcome else 'Unknown error'}. Retrying after backoff..."
        ),
    )
    def get_judgement_response(
        self, debate: DebateTotal, model: str
    ) -> Tuple[str, Dict]:
        """Gets judgment response from the API"""
        self.logger.info(f"Requesting judgment from OpenRouter for model: {model}")

        messages = [
            {
                "role": "system",
                "content": f"You are a judge. Follow these rules {self.prompts.judge_prompt}",
            },
            {"role": "user", "content": f"the debate is {debate.get_transcript()}"},
        ]

        response_data = self.client.send_request(model=model, messages=messages)

        judgment = response_data.content
        usage = {
            "completion_tokens": response_data.completion_tokens,
            "prompt_tokens": response_data.prompt_tokens,
            "total_tokens": response_data.completion_tokens
            + response_data.prompt_tokens,
        }

        return judgment, usage

    def extract_debate_result(self, xml_string: str, model: str) -> JudgeResult:
        """Extracts structured judgment result from XML response"""
        try:
            winner_matches = re.findall(r"<winnerName>(\w+)</winnerName>", xml_string)
            if not winner_matches or len(winner_matches) == 0:
                raise ValueError("Must have exactly one winner")
            winner = winner_matches[-1].lower()
            if winner not in ["opposition", "proposition"]:
                raise ValueError("Winner must be opposition or proposition")

            confidence_matches = re.findall(
                r"<confidence>(\d+)</confidence>", xml_string
            )
            if not confidence_matches or len(confidence_matches) != 1:
                raise ValueError("Must have exactly one confidence value")
            confidence = int(confidence_matches[0])
            if not 0 <= confidence <= 100:
                raise ValueError("Confidence must be between 0 and 100")

            return JudgeResult(
                model=model, winner=winner, confidence=confidence, logic=xml_string
            )

        except Exception as e:
            self.logger.error(f"Error processing judgment: {str(e)}")
            self.logger.error(f"Problematic XML: {xml_string}")

            while True:
                try:
                    winner = input("\nEnter winner (opposition/proposition): ").strip()
                    if winner not in ["opposition", "proposition"]:
                        logging.error("Invalid winner")
                        continue

                    confidence_str = input("Enter confidence (0-100): ").strip()
                    if not confidence_str.isdigit():
                        logging.error("Confidence must be a number")
                        continue

                    confidence_int = int(confidence_str)
                    if not 0 <= confidence_int <= 100:
                        logging.error("Confidence must be between 0 and 100")
                        continue

                    return JudgeResult(
                        model=model,
                        winner=winner,
                        confidence=confidence_int,
                        logic=xml_string,
                    )
                except ValueError:
                    logging.error("Invalid input, please try again")

    def process_judgment(self, debate: DebateTotal, model: str) -> JudgeResult:
        """Main method to process a judgment"""
        try:
            judgment_string, usage = self.get_judgement_response(
                debate=debate, model=model
            )

            # Track successful token usage
            debate.judge_token_counts.add_successful_call(
                model=model,
                completion_tokens=usage.get("completion_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

            judge_result = self.extract_debate_result(
                xml_string=judgment_string, model=model
            )
            debate.judge_results.append(judge_result)
            debate.save_to_json()
            return judge_result

        except Exception as e:
            # Track failed token usage if available
            if hasattr(e, "response") and hasattr(e.response, "json"):
                usage = e.response.json().get("usage", {})
                debate.judge_token_counts.add_failed_call(
                    model=model,
                    completion_tokens=usage.get("completion_tokens", 0),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )
                debate.save_to_json()
            raise
