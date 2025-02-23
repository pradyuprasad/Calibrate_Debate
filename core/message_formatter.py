from typing import List, Dict
from core.models import SpeechType, Round, DebatePrompts, DebateTotal

class MessageFormatter:
    """Formats debate state into messages for model consumption."""

    def __init__(self, prompts: DebatePrompts):
        self.prompts = prompts

    def get_chat_messages(self, debate: DebateTotal, next_round: Round) -> List[Dict]:
        """Generate chat messages for the next debate round."""
        return [
            {
                "role": "system",
                "content": self._get_system_message(next_round)
            },
            {
                "role": "user",
                "content": self._get_user_message(debate, next_round)
            }
        ]

    def _get_system_message(self, round: Round) -> str:
        """Get the system prompt for the given round."""
        prompt = {
            SpeechType.OPENING: self.prompts.first_speech_prompt,
            SpeechType.REBUTTAL: self.prompts.rebuttal_speech_prompt,
            SpeechType.CLOSING: self.prompts.final_speech_prompt
        }[round.speech_type]

        return f"You are on the {round.side.value} side. {prompt}"

    def _get_user_message(self, debate: DebateTotal, next_round: Round) -> str:
        """Format the debate motion and history for the next round."""
        return f"""The motion is: {debate.motion.topic_description}
        {self._get_debate_history(debate, next_round)}"""

    def _get_debate_history(self, debate: DebateTotal, next_round: Round) -> str:
        """Format the relevant debate history for the next round."""
        if next_round.speech_type == SpeechType.OPENING and next_round.side.value == "proposition":
            return "This is the opening speech of the debate."

        prop_speeches = debate.proposition_output.speeches
        opp_speeches = debate.opposition_output.speeches
        transcript = []

        # Add speeches up to current round
        for speech_type in SpeechType:
            # Stop if we've reached current speech type
            if speech_type == next_round.speech_type:
                # For opposition's turn, include proposition's speech if available
                if next_round.side.value == "opposition":
                    prop_speech = prop_speeches[speech_type]
                    if prop_speech != -1:
                        transcript.append(f"PROPOSITION {speech_type.value.upper()}\n{prop_speech}")
                break

            # Add both speeches from previous rounds
            prop_speech = prop_speeches[speech_type]
            opp_speech = opp_speeches[speech_type]

            if prop_speech != -1:
                transcript.append(f"PROPOSITION {speech_type.value.upper()}\n{prop_speech}")
            if opp_speech != -1:
                transcript.append(f"OPPOSITION {speech_type.value.upper()}\n{opp_speech}")

        history = "=== DEBATE HISTORY ===\n\n" + "\n\n".join(transcript) + "\n\n"
        task = f"=== YOUR TASK ===\nYou are on the {next_round.side.value} side.\n"
        task += f"You must now give your {next_round.speech_type.value} speech.\n"

        return history + task
