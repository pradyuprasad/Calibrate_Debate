import pytest

from core.message_formatter import MessageFormatter
from core.models import (DebatePrompts, DebateTopic, DebateTotal, Round, Side,
                         SpeechType)


@pytest.fixture
def sample_prompts():
    return DebatePrompts(
        first_speech_prompt="This is a first speech prompt.",
        rebuttal_speech_prompt="This is a rebuttal prompt.",
        final_speech_prompt="This is a final speech prompt.",
        judge_prompt="This is a judge prompt.",
    )


@pytest.fixture
def message_formatter(sample_prompts):
    return MessageFormatter(prompts=sample_prompts)


@pytest.fixture
def sample_debate():
    return DebateTotal(
        motion=DebateTopic(topic_description="Test topic"),
        path_to_store="test_path.json",
        proposition_model="test-model-1",
        opposition_model="test-model-2",
        prompts=DebatePrompts(
            first_speech_prompt="First speech",
            rebuttal_speech_prompt="Rebuttal speech",
            final_speech_prompt="Final speech",
            judge_prompt="Judge prompt",
        ),
    )


def test_get_system_message(message_formatter):
    opening_round = Round(side=Side.PROPOSITION, speech_type=SpeechType.OPENING)
    system_msg = message_formatter._get_system_message(opening_round)
    assert "proposition" in system_msg
    assert "This is a first speech prompt." in system_msg

    rebuttal_round = Round(side=Side.OPPOSITION, speech_type=SpeechType.REBUTTAL)
    system_msg = message_formatter._get_system_message(rebuttal_round)
    assert "opposition" in system_msg
    assert "This is a rebuttal prompt." in system_msg


def test_get_chat_messages(message_formatter, sample_debate):
    test_round = Round(side=Side.PROPOSITION, speech_type=SpeechType.OPENING)
    messages = message_formatter.get_chat_messages(sample_debate, test_round)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Test topic" in messages[1]["content"]
