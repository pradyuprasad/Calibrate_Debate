import logging
from typing import Dict, List

import requests

from core.models import APIResponse


class OpenRouterClient:
    def __init__(self, api_key: str, logger: logging.Logger):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.logger = logger

    def send_request(self, model: str, messages: List[Dict], timeout:int=300) -> APIResponse:
        """Raw API request - just sends and returns response"""
        payload = {"model": model, "messages": messages}

        self.logger.info(f"Payload is {payload}")




        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=(30, timeout))
            output = response.json()
        except requests.exceptions.Timeout as e:
            self.logger.error(f"REQUEST TIMED OUT: {str(e)}", exc_info=True)
        except ValueError as e:
            self.logger.error(f"Error decoding JSON response: {e}")
            self.logger.error(f"Response content: {response.text}")
            raise

        if "error" in output:
            raise ValueError(f"API error: {output['error']}")

        content = output["choices"][0]["message"]["content"]
        usage = output.get("usage", {})
        if not content or len(content) < 200:
            raise ValueError("Insufficient content length")
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)

        self.logger.info(f"Output is {output}")

        return APIResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
