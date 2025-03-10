import pytest
from unittest.mock import MagicMock
from core.debate_service import DebateService
from core.api_client import OpenRouterClient
from core.message_formatter import MessageFormatter
from core.models import DebateTopic, DebatePrompts, APIResponse, DebateTotal

@pytest.fixture
def mock_api_client():
    client = MagicMock(spec=OpenRouterClient)
    client.send_request.return_value = APIResponse(
        content="This is a test debate speech.",
        prompt_tokens=10,
        completion_tokens=20
    )
    return client

@pytest.fixture
def mock_formatter():
    formatter = MagicMock(spec=MessageFormatter)
    formatter.prompts = DebatePrompts(
        first_speech_prompt="First prompt",
        rebuttal_speech_prompt="Rebuttal prompt",
        final_speech_prompt="Final prompt",
        judge_prompt="Judge prompt"
    )
    formatter.get_chat_messages.return_value = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"}
    ]
    return formatter

@pytest.fixture
def debate_service(mock_api_client, mock_formatter):
    return DebateService(api_client=mock_api_client, message_formatter=mock_formatter)

def test_run_debate(debate_service, tmp_path):
    topic = DebateTopic(topic_description="Test topic")

    output_path = tmp_path / "test_debate.json"

    result = debate_service.run_debate(
        proposition_model="model1",
        opposition_model="model2",
        motion=topic,
        path_to_store=output_path
    )

    assert isinstance(result, DebateTotal)
    assert result.motion.topic_description == "Test topic"
    assert result.proposition_model == "model1"
    assert result.opposition_model == "model2"

    assert output_path.exists()

    assert debate_service.api_client.send_request.call_count == 6
