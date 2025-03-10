import pytest
from unittest.mock import MagicMock
from core.api_client import OpenRouterClient
from core.models import APIResponse

@pytest.fixture
def api_client():
    return OpenRouterClient(api_key="fake_key_for_testing")

def test_send_request_success(api_client, mocker):
    # Create a longer response that passes the minimum length check (200 chars)
    long_content = "Test response content " * 20  # This will be well over 200 characters

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": long_content}}],
        "usage": {"completion_tokens": 10, "prompt_tokens": 20}
    }

    mocker.patch('requests.post', return_value=mock_response)

    result = api_client.send_request(
        model="test-model",
        messages=[{"role": "user", "content": "test"}]
    )

    assert isinstance(result, APIResponse)
    assert result.content == long_content
    assert result.completion_tokens == 10
    assert result.prompt_tokens == 20

def test_send_request_error(api_client, mocker):
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": "Test error message"}

    mocker.patch('requests.post', return_value=mock_response)

    with pytest.raises(ValueError, match="API error: Test error message"):
        api_client.send_request(
            model="test-model",
            messages=[{"role": "user", "content": "test"}]
        )

def test_insufficient_content_length(api_client, mocker):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Short"}}],
        "usage": {"completion_tokens": 1, "prompt_tokens": 5}
    }

    mocker.patch('requests.post', return_value=mock_response)

    with pytest.raises(ValueError, match="Insufficient content length"):
        api_client.send_request(
            model="test-model",
            messages=[{"role": "user", "content": "test"}]
        )
