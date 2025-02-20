from enum import Enum
import json
from pathlib import Path
from typing import Dict, List, Union, Literal, cast
from pydantic import BaseModel, Field
import logging



class DebateTopic(BaseModel):
    topic_description: str


class Side(Enum):
    PROPOSITION = "proposition"
    OPPOSITION = "opposition"


class SpeechType(Enum):
    OPENING = "opening"
    REBUTTAL = "rebuttal"
    CLOSING = "closing"


class Round:
    def __init__(self, side: Side, speech_type: SpeechType):
        self.side = side
        self.speech_type = speech_type


class DebatePrompts(BaseModel):
    first_speech_prompt: str
    rebuttal_speech_prompt: str
    final_speech_prompt: str
    judge_prompt: str


class DebatorOutputs(BaseModel):
    side: Side
    speeches: Dict[SpeechType, Union[str, Literal[-1]]] = Field(
        default_factory=lambda: {
            speech_type: cast(Union[str, Literal[-1]], -1) for speech_type in SpeechType
        }
    )




class ModelTokenUsage(BaseModel):
    successful_calls: int = 0
    failed_calls: int = 0
    successful_completion_tokens: int = 0
    successful_prompt_tokens: int = 0
    successful_total_tokens: int = 0
    failed_completion_tokens: int = 0
    failed_prompt_tokens: int = 0
    failed_total_tokens: int = 0

    @property
    def total_completion_tokens(self) -> int:
        return self.successful_completion_tokens + self.failed_completion_tokens

    @property
    def total_prompt_tokens(self) -> int:
        return self.successful_prompt_tokens + self.failed_prompt_tokens

    @property
    def total_tokens(self) -> int:
        return self.successful_total_tokens + self.failed_total_tokens


class TokenCount(BaseModel):
    model_usages: Dict[str, ModelTokenUsage] = Field(default_factory=dict)

    def add_successful_call(
        self, model: str, completion_tokens: int, prompt_tokens: int, total_tokens: int
    ):
        if model not in self.model_usages:
            self.model_usages[model] = ModelTokenUsage()

        usage = self.model_usages[model]
        usage.successful_calls += 1
        usage.successful_completion_tokens += completion_tokens
        usage.successful_prompt_tokens += prompt_tokens
        usage.successful_total_tokens += total_tokens

    def add_failed_call(
        self, model: str, completion_tokens: int, prompt_tokens: int, total_tokens: int
    ):
        if model not in self.model_usages:
            self.model_usages[model] = ModelTokenUsage()

        usage = self.model_usages[model]
        usage.failed_calls += 1
        usage.failed_completion_tokens += completion_tokens
        usage.failed_prompt_tokens += prompt_tokens
        usage.failed_total_tokens += total_tokens




class JudgeResult(BaseModel):
    model: str
    winner: Literal['opposition', 'proposition']
    confidence: int = Field(ge=0, le=100)
    logic: str



class DebateTotal(BaseModel):
    motion: DebateTopic
    path_to_store: Path
    proposition_model: str
    opposition_model: str
    prompts: DebatePrompts
    proposition_output: DebatorOutputs = Field(
        default_factory=lambda: DebatorOutputs(side=Side.PROPOSITION)
    )
    opposition_output: DebatorOutputs = Field(
        default_factory=lambda: DebatorOutputs(side=Side.OPPOSITION)
    )
    judge_models: List[str] = Field(default_factory=list)
    judge_results: List[JudgeResult] = Field(default_factory=list)
    debator_token_counts: TokenCount = Field(default_factory=TokenCount)
    judge_token_counts: TokenCount = Field(default_factory=TokenCount)

    def to_dict(self) -> dict:
        return {
            "motion": {
                "topic_description": self.motion.topic_description,
            },
            "path_to_store": str(self.path_to_store),
            "proposition_model": self.proposition_model,
            "opposition_model": self.opposition_model,
            "prompts": self.prompts.dict(),
            "proposition_output": {
                "side": self.proposition_output.side.value,
                "speeches": {k.value: v for k, v in self.proposition_output.speeches.items()}
            },
            "opposition_output": {
                "side": self.opposition_output.side.value,
                "speeches": {k.value: v for k, v in self.opposition_output.speeches.items()}
            },
            "judge_models": self.judge_models,
            "judge_results": [result.dict() for result in self.judge_results],
            "debator_token_counts": {
                model: usage.dict()
                for model, usage in self.debator_token_counts.model_usages.items()
            },
            "judge_token_counts" : {
                model: usage.model_dump()
                for model, usage in self.judge_token_counts.model_usages.items()
            }

        }

    def save_to_json(self) -> None:
        data = self.to_dict()
        with open(self.path_to_store, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved to {self.path_to_store}")

    @classmethod
    def load_from_json(cls, path: Union[str, Path]) -> 'DebateTotal':
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)

        data['motion'] = DebateTopic(
            topic_description=data['motion']['topic_description']
        )

        data['prompts'] = DebatePrompts(**data['prompts'])

        data['proposition_output'] = DebatorOutputs(
            side=Side(data['proposition_output']['side']),
            speeches={SpeechType(k): v for k, v in data['proposition_output']['speeches'].items()}
        )

        data['opposition_output'] = DebatorOutputs(
            side=Side(data['opposition_output']['side']),
            speeches={SpeechType(k): v for k, v in data['opposition_output']['speeches'].items()}
        )

        data['judge_results'] = [JudgeResult(**result) for result in data['judge_results']]

        debator_token_counts = TokenCount()
        for model, usage_data in data.get('debator_token_counts', {}).items():
            model_usage = ModelTokenUsage(**usage_data)
            debator_token_counts.model_usages[model] = model_usage


        judge_token_counts = TokenCount()
        for model, usage_data in data.get('judge_token_counts', {}).items():
            model_usage = ModelTokenUsage(**usage_data)
            judge_token_counts.model_usages[model] = model_usage

        data['debator_token_counts'] = debator_token_counts
        data['judge_token_counts'] = judge_token_counts

        data['path_to_store'] = Path(data['path_to_store'])

        return cls(**data)

    def get_transcript(self) -> Dict[str, str]:
        transcript = {}
        transcript["Proposition Opening Speech"] = self.proposition_output.speeches[SpeechType.OPENING]
        transcript["Opposition Opening Speech"] = self.opposition_output.speeches[SpeechType.OPENING]
        transcript["Proposition Rebuttal"] = self.proposition_output.speeches[SpeechType.REBUTTAL]
        transcript["Opposition Rebuttal"] = self.opposition_output.speeches[SpeechType.REBUTTAL]
        transcript["Proposition Closing Speech"] = self.proposition_output.speeches[SpeechType.CLOSING]
        transcript["Opposition Closing Speech"] = self.opposition_output.speeches[SpeechType.CLOSING]

        missing_speeches = [speech for speech, content in transcript.items() if content == -1]
        if missing_speeches:
            raise ValueError(f"Debate is incomplete. Missing speeches: {', '.join(missing_speeches)}")

        return cast(Dict[str, str], transcript)
