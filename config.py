from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv
from core.models import DebatePrompts, BetPatternConfig
import yaml
from core.message_formatter import MessageFormatter
from core.api_client import OpenRouterClient
from core.debate_service import DebateService
from core.judgement_processor import JudgementProcessor
from core.logger import LoggerFactory


@dataclass
class Config:
    ai_models_dir: Path = Path("ai_models")
    debate_models_list_path: Path = field(init=False)
    judge_models_list_path: Path = field(init=False)
    topic_dir: Path = Path("topics")
    topic_list_path: Path = field(init=False)
    prompts_dir: Path = Path("prompts")
    prompts_path_yaml: Path = field(init=False)

    outputs_dir: Path = Path("outputs")
    debates_dir: Path = field(init=False)
    judgments_dir: Path = field(init=False)

    samples_dir: Path = Path("samples")
    sample_debates_dir: Path = field(init=False)
    sample_judgments_dir: Path = field(init=False)

    tournament_dir: Path = Path("tournament")
    tournament_results_path: Path = field(init=False)
    num_rounds: int = 3
    k_factor: int = 64

    api_key: str = field(init=False)
    prompts: DebatePrompts = field(init=False)
    message_formatter: MessageFormatter = field(init=False)
    api_client: OpenRouterClient = field(init=False)
    debate_service: DebateService = field(init=False)
    judgement_processor: JudgementProcessor = field(init=False)
    bet_pattern_config: BetPatternConfig = field(init=False)


    logger = LoggerFactory()

    def load_debate_prompts(self) -> DebatePrompts:
        with open(self.prompts_path_yaml, "r") as file:
            prompts = yaml.safe_load(file)

        debator_prompts = DebatePrompts(
            first_speech_prompt=prompts["first_speech"],
            rebuttal_speech_prompt=prompts["rebuttal_speech"],
            final_speech_prompt=prompts["final_speech"],
            judge_prompt=prompts["judging_prompt"],
        )

        return debator_prompts

    def __post_init__(self):
        load_dotenv()
        self.ai_models_dir.mkdir(exist_ok=True)
        self.topic_dir.mkdir(exist_ok=True)
        self.prompts_dir.mkdir(exist_ok=True)

        self.outputs_dir.mkdir(exist_ok=True)
        self.debates_dir = self.outputs_dir / "debates"
        self.judgments_dir = self.outputs_dir / "judgments"
        self.debates_dir.mkdir(exist_ok=True)
        self.judgments_dir.mkdir(exist_ok=True)

        self.samples_dir.mkdir(exist_ok=True)
        self.sample_debates_dir = self.samples_dir / "debates"
        self.sample_judgments_dir = self.samples_dir / "judgments"
        self.sample_debates_dir.mkdir(exist_ok=True)
        self.sample_judgments_dir.mkdir(exist_ok=True)

        self.tournament_dir.mkdir(exist_ok=True)
        self.tournament_results_path = self.tournament_dir / "tournament_results.json"

        self.debate_models_list_path = self.ai_models_dir / "debate_models.json"
        self.judge_models_list_path = self.ai_models_dir / "judge_models.json"
        self.topic_list_path = self.topic_dir / "topics_list.json"
        self.prompts_path_yaml = self.prompts_dir / "debate_prompts.yaml"

        self.bet_pattern_config = BetPatternConfig(
            bet_amount_xml_tag="bet_amount",
            bet_logic_private_xml_tag="bet_logic_private",
        )

        self.api_key = os.environ["OPENROUTER_API_KEY"]
        if not self.api_key:
            raise RuntimeError("No OPENROUTER_API_KEY found")
        self.prompts = self.load_debate_prompts()
        self.message_formatter = MessageFormatter(
            prompts=self.prompts, bet_pattern_config=self.bet_pattern_config
        )
        self.api_client = OpenRouterClient(api_key=self.api_key, logger=self.logger.get_logger())
        self.debate_service = DebateService(
            api_client=self.api_client,
            message_formatter=self.message_formatter,
            bet_pattern_config=self.bet_pattern_config,
            logger=self.logger.get_logger()
        )
        self.judgement_processor = JudgementProcessor(
            prompts=self.prompts, client=self.api_client,
            logger=self.logger.get_logger()
        )
