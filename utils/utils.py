from collections import Counter
from typing import List

from core.models import Round, Side, SpeechType


def make_rounds() -> List[Round]:
    """
    Creates standard debate format rounds.
    Ensures each side has exactly one of each speech type.
    """
    rounds = [
        (Side.PROPOSITION, SpeechType.OPENING),
        (Side.OPPOSITION, SpeechType.OPENING),
        (Side.PROPOSITION, SpeechType.REBUTTAL),
        (Side.OPPOSITION, SpeechType.REBUTTAL),
        (Side.PROPOSITION, SpeechType.CLOSING),
        (Side.OPPOSITION, SpeechType.CLOSING),
    ]

    # Validate no duplicates per side/speech type
    speech_counts = Counter((side, speech_type) for side, speech_type in rounds)
    for count in speech_counts.values():
        if count > 1:
            raise ValueError("Found duplicate speech type for a side in rounds")

    return [Round(side, speech_type) for side, speech_type in rounds]
