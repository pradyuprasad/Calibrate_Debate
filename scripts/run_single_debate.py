from core.debate_service import DebateService
from core.api_client import OpenRouterClient
from core.message_formatter import MessageFormatter
from prompts.load_prompts import get_debate_prompt
from core.judgement_processor import JudgementProcessor
from config import Config
from topics.load_topics import load_topics
from core.models import DebateTotal

from dotenv import load_dotenv
import os

load_dotenv()
config = Config()
api_key = os.environ['OPENROUTER_API_KEY']
api_client = OpenRouterClient(api_key=api_key)
prompt = get_debate_prompt(config=config)
message_formatter = MessageFormatter(prompts=prompt)
motion = load_topics(config)[0]

debate_service = DebateService(api_client=api_client, message_formatter=message_formatter)

judgement_processor = JudgementProcessor(prompts=prompt, client=api_client)

debate = DebateTotal.load_from_json('test_debate_new.json')

judgement_processor.process_judgment(debate=debate, model="openai/o1-mini")
