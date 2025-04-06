from pydantic import BaseModel
from typing import Dict, List

class DebateData(BaseModel):
    """Standardized format for debate data."""
    id: str
    round: int
    tournament: str
    path: str
    proposition_model: str
    opposition_model: str
    topic: str
    winner: str
    judge_results: Dict[str, int]
    prop_bets: Dict[str, float]
    opp_bets: Dict[str, float]
    judge_agreement: str

class ModelStats(BaseModel):
    """Statistics for a single model."""
    wins: int = 0
    losses: int = 0
    prop_wins: int = 0
    prop_losses: int = 0
    opp_wins: int = 0
    opp_losses: int = 0
    debates: int = 0
    win_rate: float = 0.0
    prop_win_rate: float = 0.0
    opp_win_rate: float = 0.0

class TopicStats(BaseModel):
    """Statistics for a single debate topic."""
    count: int = 0
    prop_wins: int = 0
    opp_wins: int = 0
    unanimous_decisions: int = 0
    split_decisions: int = 0
    models: List[str] = []
    prop_win_rate: float = 0.0
    unanimous_rate: float = 0.0

class DebateResults(BaseModel):
    """Complete debate analysis results."""
    debates: List[DebateData]
    model_stats: Dict[str, ModelStats]
    topic_stats: Dict[str, TopicStats]
