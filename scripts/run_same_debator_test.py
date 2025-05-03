#!/usr/bin/env python3
"""
Simple script to run a debate where a model debates against itself.
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

from config import Config
from core.models import DebateType
from topics.load_topics import load_topics

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
config = Config()

# Get a topic
topics = load_topics(config)
topic = topics[0]
logger.info(f"Using topic: {topic.topic_description}")

# Set up a debate with the model debating itself
output_path = Path("sample_same_debator_debate.json")
debate_type = DebateType.SAME_DEBATOR
model = "anthropic/claude-3.5-sonnet"  # Using Claude as an example
logger.info(f"Starting same-debator debate with model: {model}")

debate = config.debate_service.run_debate(
    proposition_model=model,
    opposition_model=model,  # Same model for both sides
    motion=topic,
    path_to_store=output_path,
    debate_type=debate_type,
)

print(f"Debate completed and saved to {output_path}")
print("Bets placed:")
assert debate.debator_bets is not None
for bet in debate.debator_bets:
    print(f"{bet.side.value} {bet.speech_type.value}: {bet.amount}")