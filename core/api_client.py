from typing import Dict, List
from core.models import APIResponse
import requests
import logging


class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.logger = logging.getLogger(self.__class__.__name__)




    def send_request(self, model: str, messages: List[Dict]) -> APIResponse:
        """Raw API request - just sends and returns response"""
        payload = {
            "model": model,
            "messages": messages
        }

        self.logger.info(f"Payload is {payload}")


        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload
        )

        try:
            output = response.json()
        except ValueError as e:
            self.logger.error(f"Error decoding JSON response: {e}")
            self.logger.error(f"Response content: {response.text}")
            raise

        if "error" in response:
                raise ValueError(f"API error: {response['error']}")

        content = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        if not content or len(content) < 200:
                raise ValueError("Insufficient content length")
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens=usage.get("prompt_tokens", 0)



        self.logger.info(f"Output is {output}")

        return APIResponse(content=content, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
