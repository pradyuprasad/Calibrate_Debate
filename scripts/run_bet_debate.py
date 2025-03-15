#!/usr/bin/env python3
"""
Simple script to run a sample debate with confidence betting.
"""

import logging
from pathlib import Path
from dotenv import load_dotenv
from config import Config
from topics.load_topics import load_topics
from core.models import DebateType

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

# Set up a debate with private betting
output_path = Path("sample_bet_debate2.json")
debate_type = DebateType.PUBLIC_BET
logger.info(f"Starting debate with type: {debate_type.value}")
logger.info("Proposition: anthropic/claude-3.5-sonnet")
logger.info("Opposition: google/gemma-3-27b-it:free")

debate = config.debate_service.run_debate(
    proposition_model="anthropic/claude-3.5-sonnet",
    opposition_model="google/gemma-3-27b-it:free",
    motion=topic,
    path_to_store=output_path,
    debate_type=debate_type,
)

print(f"Debate completed and saved to {output_path}")
print("Bets placed:")
assert debate.debator_bets is not None
for bet in debate.debator_bets:
    print(f"{bet.side.value} {bet.speech_type.value}: {bet.amount}")
