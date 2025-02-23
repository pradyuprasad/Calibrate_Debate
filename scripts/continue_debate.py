from pathlib import Path
from core.models import DebateTotal, Side, SpeechType
from core.debate import get_judgement
from scripts.analyse_sample_debates import checkIfComplete
import logging
import requests
import os
from dotenv import load_dotenv
from config import Config


def get_valid_response(messages: list, model: str) -> tuple[str, dict]:
    """
    Sends request to OpenRouter API and gets response
    Returns tuple of (speech_text, response_json)
    """
    logging.info(f"Starting API request to OpenRouter for model: {model}")
    logging.info(f"Request messages: {messages}")

    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "provider": {
            "ignore": []
        }
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )

    response_json = response.json()
    logging.info(f"Raw API response: {response_json}")

    if response.status_code != 200:
        error_msg = f"API returned error: {response_json.get('error', {}).get('message')}"
        logging.error(error_msg)
        raise ValueError(error_msg)

    speech = response_json["choices"][0]["message"]["content"]

    if not speech or len(speech) < 100:
        error_msg = "API returned empty speech content"
        logging.error(error_msg)
        raise ValueError(error_msg)

    return speech, response_json

def continue_debate(debate_path: Path) -> None:
    """
    Continues a partially completed debate by completing any missing speeches
    and judge evaluations.

    Args:
        debate_path: Path to the JSON file containing the partial debate
    """
    # Load the existing debate
    debate = DebateTotal.load_from_json(debate_path)

    # Check each side's speeches
    for side, output in [(Side.PROPOSITION, debate.proposition_output),
                        (Side.OPPOSITION, debate.opposition_output)]:

        model = debate.proposition_model if side == Side.PROPOSITION else debate.opposition_model

        for speech_type in SpeechType:
            if output.speeches[speech_type] == -1:
                logging.info(f"Generating missing {side.value} {speech_type.value} speech")

                # Get context of previous speeches for prompting
                context = []
                for prev_type in SpeechType:
                    if prev_type == speech_type:
                        break
                    prop_speech = debate.proposition_output.speeches[prev_type]
                    opp_speech = debate.opposition_output.speeches[prev_type]
                    if prop_speech != -1:
                        context.append({
                            "role": "assistant" if side == Side.OPPOSITION else "user",
                            "content": f"Proposition speech: {prop_speech}"
                        })
                    if opp_speech != -1:
                        context.append({
                            "role": "assistant" if side == Side.PROPOSITION else "user",
                            "content": f"Opposition speech: {opp_speech}"
                        })

                # Get appropriate prompt template
                prompt = {
                    SpeechType.OPENING: debate.prompts.first_speech_prompt,
                    SpeechType.REBUTTAL: debate.prompts.rebuttal_speech_prompt,
                    SpeechType.CLOSING: debate.prompts.final_speech_prompt
                }[speech_type]

                # Generate the speech
                messages = [
                    {
                        "role": "system",
                        "content": f"You are on the {side.value} side. {prompt}"
                    },
                    {
                        "role": "user",
                        "content": f"You are debating {debate.motion.topic_description}"
                    },
                    *context
                ]

                try:
                    speech, response_json = get_valid_response(messages, model)

                    # Update the debate state
                    output.speeches[speech_type] = speech

                    # Track token usage
                    usage = response_json.get("usage", {})
                    debate.debator_token_counts.add_successful_call(
                        model=model,
                        completion_tokens=usage.get("completion_tokens", 0),
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0)
                    )

                    # Save after each speech
                    debate.save_to_json()

                except Exception as e:
                    logging.error(f"Error generating speech: {e}")
                    if hasattr(e, 'response') and hasattr(e.response, 'json'):
                        usage = e.response.json().get('usage', {})
                        debate.debator_token_counts.add_failed_call(
                            model=model,
                            completion_tokens=usage.get("completion_tokens", 0),
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            total_tokens=usage.get("total_tokens", 0)
                        )
                    raise

    # Run judge evaluations if needed
    if debate.judge_models and not debate.judge_results:
        for judge_model in debate.judge_models:
            get_judgement(debate=debate, prompts=debate.prompts, judge_model=judge_model)
            debate.save_to_json()

    logging.info("Debate continuation completed")

if __name__ == "__main__":
# Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load environment variables (make sure OPENROUTER_API_KEY is set)
    load_dotenv()
    config = Config()
    paths = config.sample_debates_dir.glob("*.json")
    for path in paths:
        if not checkIfComplete(path):
            print(f"{path} is not Complete")

    for path in paths:
        continue_debate(path)
