from config import Config
from core.models import DebateTopic
from typing import List
import json


def load_topics(config: Config) -> List[DebateTopic]:
    with open(config.topic_list_path) as f:
        data = json.load(f)

    topic_list = []
    for topic in data:
        topic_list.append(DebateTopic(**topic))

    return topic_list
