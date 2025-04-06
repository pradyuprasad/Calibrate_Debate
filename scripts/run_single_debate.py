from dotenv import load_dotenv

from config import Config
from core.models import DebateTotal
from topics.load_topics import load_topics

load_dotenv()
config = Config()
api_client = config.api_client
prompt = config.prompts
message_formatter = config.message_formatter
motion = load_topics(config)[0]

debate_service = config.debate_service

judgement_processor = config.judgement_processor

debate = DebateTotal.load_from_json("test_debate_new.json")

judgement_processor.process_judgment(debate=debate, model="openai/o1-mini")
