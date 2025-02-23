from typing import Dict, List

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




    def send_request(self, model: str, messages: List[Dict]) -> Dict:
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

        try:  # Added a try-except block, to handle potential JSON decoding errors
            output = response.json()
        except ValueError as e:
            self.logger.error(f"Error decoding JSON response: {e}")
            self.logger.error(f"Response content: {response.text}")
            return {"error": "Failed to decode JSON response"}
            raise


        self.logger.info(f"Output is {output}")

        return output
