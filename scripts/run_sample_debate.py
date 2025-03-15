import logging
import random
from pathlib import Path
from config import Config
from core.debate_service import DebateService
from topics.load_topics import load_topics
from core.models import DebateTopic, DebateType
from core.judgement_processor import JudgementProcessor
from ai_models.load_models import load_debate_models, load_judge_models
from typing import Dict, List, Tuple
from scripts.utils import sanitize_model_name

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment

def create_debate_pairs(models_dict: Dict[str, List[float]]) -> List[Tuple[str, str]]:
   # Convert dict keys to list of model names
   models = list(models_dict.keys())


   random.shuffle(models)
   pairs = []
   for i in range(0, len(models)-1, 2):
       if i+1 < len(models):
           pairs.append((models[i], models[i+1]))


   return pairs



def run_single_sample_debate(debate_service: DebateService, proposition_model:str, opposition_model: str, judge_models: List[str], path_to_use:Path, judgment_processor: JudgementProcessor, topic: DebateTopic) -> None:
    debate_obj = debate_service.run_debate(proposition_model=proposition_model, opposition_model=opposition_model, path_to_store=path_to_use,debate_type=DebateType.BASELINE, motion=topic)

    for model in judge_models:
        for _ in range(3):
            judgment_processor.process_judgment(debate=debate_obj, model=model)

def run_all_sample_debates() -> None:
    config = Config()
    models_dict = load_debate_models(config)
    pair_list = create_debate_pairs(models_dict)
    topics = load_topics(config)
    judge_models = list(load_judge_models(config).keys())
    debate_service = config.debate_service
    judgement_processor = config.judgement_processor

    for idx, (proposition, opposition) in enumerate(pair_list):
        logging.info(f"running debate {idx+1}/{len(pair_list)} with proposition {proposition} and opposition {opposition}")

        clean_prop_name = sanitize_model_name(proposition)
        clean_opp_name = sanitize_model_name(opposition)

        path_to_use = config.sample_debates_dir / f"{clean_prop_name}_{clean_opp_name}.json"
        topic_idx = idx % len(topics)
        topic_to_use = topics[topic_idx]

        run_single_sample_debate(
            debate_service=debate_service,
            proposition_model=proposition,
            opposition_model=opposition,
            judge_models=judge_models,
            judgment_processor=judgement_processor,
            topic=topic_to_use,
            path_to_use=path_to_use
        )

run_all_sample_debates()
