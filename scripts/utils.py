import logging
from pathlib import Path

from core.models import DebateTotal, SpeechType


def sanitize_model_name(model_name: str) -> str:
    """Convert model name to a valid filename by replacing / with _"""
    return model_name.replace('/', '_')

def checkIfComplete(path: Path) -> bool:
    try:
        # Load the debate from the json file
        debate = DebateTotal.load_from_json(path)

        # Check all speeches in both proposition and opposition outputs
        for speech_type in SpeechType:
            # Check proposition speeches
            if debate.proposition_output.speeches[speech_type] == -1:
                return False

            # Check opposition speeches
            if debate.opposition_output.speeches[speech_type] == -1:
                return False

        return True

    except Exception as e:
        # Handle any errors in loading or processing the file
        logging.error(f"Error checking debate completion status: {str(e)}")
        return False
