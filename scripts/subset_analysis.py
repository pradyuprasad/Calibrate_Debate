#!/usr/bin/env python3
"""
debate_analysis.py - Comprehensive debate analysis with subset breakdowns
Analyzes calibration, confidence patterns, and performance metrics across different subsets
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from core.models import DebateTotal, Side, SpeechType


# Data structures
@dataclass
class ConfidenceMetrics:
    confidence: float
    won: bool
    side: str
    model: str
    agreement_level: str

@dataclass
class DebateMetrics:
    proposition_model: str
    opposition_model: str
    winner: str
    prop_confidence: float
    opp_confidence: float
    agreement_level: str
    prop_votes: int
    opp_votes: int
    total_judges: int
    prop_opening_conf: float
    prop_closing_conf: float
    opp_opening_conf: float
    opp_closing_conf: float

@dataclass
class CalibrationStats:
    total_predictions: int
    wins: int
    win_rate: float
    avg_confidence: float
    calibration_error: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_significance: Dict[str, float]

# Constants
CONFIDENCE_BINS = [(0, 25), (26, 50), (51, 75), (76, 100)]
DETAILED_BINS = [(i, i+5) for i in range(50, 96, 5)]
AGREEMENT_THRESHOLDS = {
    "high": 0.8,
    "medium": 0.6,
    "low": 0.0
}

def calculate_aggregate_statistics(data: Dict) -> None:
    """Calculate comprehensive aggregate statistics across all tracked metrics"""

    # Overall win rates and confidence stats
    for side in ["proposition", "opposition"]:
        conf_data = data["confidence_data"][side]
        wins = sum(1 for d in conf_data if d["won"])
        total = len(conf_data)

        data[f"{side}_stats"] = {
            "win_rate": (wins / total) if total > 0 else 0,
            "avg_confidence": sum(d["confidence"] for d in conf_data) / total if total > 0 else 0,
            "total_debates": total,
            "wins": wins
        }

    # Model-specific aggregate stats
    for model in data["model_stats"]:
        model_data = data["model_stats"][model]
        total_debates = len(model_data["wins"])

        model_data["aggregate"] = {
            "total_debates": total_debates,
            "overall_win_rate": sum(model_data["wins"]) / total_debates if total_debates > 0 else 0,
            "avg_confidence": sum(model_data["confidences"]) / total_debates if total_debates > 0 else 0,
            "prop_win_rate": sum(1 for p in model_data["proposition_performances"] if p["won"]) /
                           len(model_data["proposition_performances"]) if model_data["proposition_performances"] else 0,
            "opp_win_rate": sum(1 for p in model_data["opposition_performances"] if p["won"]) /
                          len(model_data["opposition_performances"]) if model_data["opposition_performances"] else 0
        }

    # Agreement level stats
    total_debates = sum(data["agreement_stats"].values())
    data["agreement_summary"] = {
        level: {
            "count": count,
            "percentage": (count / total_debates * 100) if total_debates > 0 else 0
        }
        for level, count in data["agreement_stats"].items()
    }

    # Calibration errors
    for side in ["proposition", "opposition"]:
        conf_data = data["confidence_data"][side]
        if conf_data:
            actual_win_rate = data[f"{side}_stats"]["win_rate"] * 100
            avg_confidence = data[f"{side}_stats"]["avg_confidence"]
            data[f"{side}_stats"]["calibration_error"] = abs(avg_confidence - actual_win_rate)

    # Confidence bin statistics
    for bin_range in CONFIDENCE_BINS:
        low, high = bin_range
        bin_key = f"{low}-{high}"

        for side in ["proposition", "opposition"]:
            bin_data = [d for d in data["confidence_data"][side]
                       if low <= d["confidence"] <= high]

            if bin_data:
                wins = sum(1 for d in bin_data if d["won"])
                total = len(bin_data)
                data["bin_stats"][bin_key][side] = {
                    "win_rate": (wins / total) * 100,
                    "count": total,
                    "calibration_error": abs((wins / total * 100) - ((low + high) / 2))
                }

def wilson_score_interval(wins: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Calculate Wilson score interval for binomial proportion"""
    if n == 0:
        return 0, 0, 0

    p = wins / n
    denominator = 1 + z**2/n
    center = (p + z**2/(2*n))/denominator
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))/denominator

    return p, max(0, center - spread), min(1, center + spread)

def load_debate_data() -> Dict:
    """
    Load and process all debate data, with extensive validation and metrics tracking
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("debate_analysis")

    tournament_dirs = [
        Path("tournament/bet_tournament_20250316_1548"),
        Path("tournament/bet_tournament_20250317_1059")
    ]

    # Load tournament results and handle model exclusions
    combined_model_stats = {}
    for tournament_dir in tournament_dirs:
        try:
            with open(tournament_dir / "tournament_results.json", "r") as f:
                tournament_results = json.load(f)
                combined_model_stats.update(tournament_results.get("model_stats", {}))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading tournament results from {tournament_dir}: {e}")

    # Handle model exclusions
    all_models = list(combined_model_stats.keys())
    print("\n=== AVAILABLE MODELS ===")
    for i, model in enumerate(all_models):
        print(f"[{i}] {model}")

    excluded_models = []
    if input("\nExclude models? (y/n): ").lower().strip() == 'y':
        indices = input("Enter indices to exclude (comma-separated): ").strip()
        try:
            excluded_models = [all_models[int(i)] for i in indices.split(',') if i.strip()]
            print(f"Excluding models: {', '.join(excluded_models)}")
        except (ValueError, IndexError):
            logger.warning("Invalid indices provided, no models excluded")

    # Initialize comprehensive tracking structures
    debate_data = {
        "metrics": [],
        "confidence_data": defaultdict(list),
        "model_stats": defaultdict(lambda: defaultdict(list)),
        "topic_stats": defaultdict(lambda: defaultdict(list)),
        "agreement_stats": defaultdict(int),
        "temporal_stats": defaultdict(list)
    }

    # Process all debates
    for tournament_dir in tournament_dirs:
        for round_dir in tournament_dir.glob("round_*"):
            round_num = int(round_dir.name.split("_")[1])

            for debate_path in round_dir.glob("*.json"):
                try:
                    debate = DebateTotal.load_from_json(debate_path)

                    # Skip debates with excluded models
                    if debate.proposition_model in excluded_models or debate.opposition_model in excluded_models:
                        continue

                    metrics = process_single_debate(debate, round_num)
                    if metrics:
                        update_tracking_structures(debate_data, metrics)

                except Exception as e:
                    logger.error(f"Error processing debate {debate_path}: {e}")

    # Calculate aggregate statistics
    calculate_aggregate_statistics(debate_data)

    return debate_data

def process_single_debate(debate: DebateTotal, round_num: int) -> Optional[DebateMetrics]:
    """Process a single debate with comprehensive metric extraction"""
    if not debate.debator_bets or not debate.judge_results:
        return None

    # Extract votes and determine winner
    prop_votes = sum(1 for r in debate.judge_results if r.winner == "proposition")
    opp_votes = sum(1 for r in debate.judge_results if r.winner == "opposition")
    total_judges = len(debate.judge_results)
    winner = "proposition" if prop_votes > opp_votes else "opposition"

    # Calculate agreement level
    agreement_ratio = max(prop_votes, opp_votes) / total_judges
    agreement_level = next(
        level for level, threshold in AGREEMENT_THRESHOLDS.items()
        if agreement_ratio >= threshold
    )

    # Extract confidence scores
    confidence_scores = defaultdict(lambda: defaultdict(float))
    for bet in debate.debator_bets:
        confidence_scores[bet.side][bet.speech_type] = bet.amount

    return DebateMetrics(
        proposition_model=debate.proposition_model,
        opposition_model=debate.opposition_model,
        winner=winner,
        prop_confidence=confidence_scores[Side.PROPOSITION][SpeechType.OPENING],
        opp_confidence=confidence_scores[Side.OPPOSITION][SpeechType.OPENING],
        agreement_level=agreement_level,
        prop_votes=prop_votes,
        opp_votes=opp_votes,
        total_judges=total_judges,
        prop_opening_conf=confidence_scores[Side.PROPOSITION][SpeechType.OPENING],
        prop_closing_conf=confidence_scores[Side.PROPOSITION][SpeechType.CLOSING],
        opp_opening_conf=confidence_scores[Side.OPPOSITION][SpeechType.OPENING],
        opp_closing_conf=confidence_scores[Side.OPPOSITION][SpeechType.CLOSING]
    )

def update_tracking_structures(data: Dict, metrics: DebateMetrics) -> None:
    """Update all tracking structures with debate metrics"""
    data["metrics"].append(metrics)

    # Update confidence tracking
    for side, conf in [("proposition", metrics.prop_confidence), ("opposition", metrics.opp_confidence)]:
        data["confidence_data"][side].append({
            "confidence": conf,
            "won": metrics.winner == side,
            "agreement_level": metrics.agreement_level
        })

    # Update model tracking
    for model, side, conf, won in [
        (metrics.proposition_model, "proposition", metrics.prop_confidence, metrics.winner == "proposition"),
        (metrics.opposition_model, "opposition", metrics.opp_confidence, metrics.winner == "opposition")
    ]:
        data["model_stats"][model]["confidences"].append(conf)
        data["model_stats"][model]["wins"].append(won)
        data["model_stats"][model][f"{side}_performances"].append({
            "confidence": conf,
            "won": won,
            "agreement_level": metrics.agreement_level
        })

    # Update agreement stats
    data["agreement_stats"][metrics.agreement_level] += 1
