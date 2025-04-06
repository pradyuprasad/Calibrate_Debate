from collections import defaultdict

from config import Config
from core.models import DebateTotal


def analyze_judge_consistency(sample_debates_list):
    # Track all judgments and consensus data
    all_judgments = []
    model_votes = defaultdict(list)
    debate_consensus = {}

    # Process all debates and collect judgments
    for debate_path in sample_debates_list:
        debate_object = DebateTotal.load_from_json(debate_path)
        debate_id = debate_path.stem

        # Skip if no judgments
        if not debate_object.judge_results:
            continue

        # Store all judgments for this debate
        debate_judgments = []
        for judgment in debate_object.judge_results:
            model_name = judgment.model
            winner = judgment.winner
            confidence = judgment.confidence / 100.0  # Normalize to 0-1

            # Store judgment details
            judgment_data = {
                "debate_id": debate_id,
                "model": model_name,
                "winner": winner,
                "confidence": confidence,
            }

            debate_judgments.append(judgment_data)
            all_judgments.append(judgment_data)
            model_votes[model_name].append(judgment_data)

        # Calculate debate consensus
        prop_votes = sum(1 for j in debate_judgments if j["winner"] == "proposition")
        opp_votes = len(debate_judgments) - prop_votes

        # Also calculate confidence-weighted consensus
        prop_confidence = sum(
            j["confidence"] for j in debate_judgments if j["winner"] == "proposition"
        )
        opp_confidence = sum(
            j["confidence"] for j in debate_judgments if j["winner"] == "opposition"
        )

        # Store consensus data
        debate_consensus[debate_id] = {
            "vote_count": {"proposition": prop_votes, "opposition": opp_votes},
            "confidence_weighted": {
                "proposition": prop_confidence,
                "opposition": opp_confidence,
            },
            "simple_consensus": "proposition"
            if prop_votes > opp_votes
            else "opposition",
            "weighted_consensus": "proposition"
            if prop_confidence > opp_confidence
            else "opposition",
            "judgments": debate_judgments,
        }

    # Calculate model consistency metrics
    model_consistency = {}
    for model_name, votes in model_votes.items():
        disagreements = 0
        total_votes = len(votes)
        total_confidence = 0
        confidence_correct_sum = 0
        confidence_incorrect_sum = 0

        for vote in votes:
            debate_id = vote["debate_id"]
            consensus = debate_consensus[debate_id]["simple_consensus"]
            confidence = vote["confidence"]
            total_confidence += confidence

            # Count disagreements with consensus
            if vote["winner"] != consensus:
                disagreements += 1
                # Track confidence on incorrect votes
                confidence_incorrect_sum += confidence
            else:
                # Track confidence on correct votes
                confidence_correct_sum += confidence

        avg_confidence = total_confidence / total_votes if total_votes > 0 else 0
        avg_confidence_when_correct = (
            confidence_correct_sum / (total_votes - disagreements)
            if (total_votes - disagreements) > 0
            else 0
        )
        avg_confidence_when_incorrect = (
            confidence_incorrect_sum / disagreements if disagreements > 0 else 0
        )

        # Calculate confidence-weighted agreement rate
        if total_confidence > 0:
            confidence_weighted_agreement = 1 - (
                confidence_incorrect_sum / total_confidence
            )
        else:
            confidence_weighted_agreement = 0

        model_consistency[model_name] = {
            "total_votes": total_votes,
            "disagreements": disagreements,
            "agreement_rate": 1 - (disagreements / total_votes)
            if total_votes > 0
            else 0,
            "confidence_weighted_agreement": confidence_weighted_agreement,
            "avg_confidence": avg_confidence,
            "avg_confidence_when_correct": avg_confidence_when_correct,
            "avg_confidence_when_incorrect": avg_confidence_when_incorrect,
            "calibration_gap": avg_confidence_when_correct
            - avg_confidence_when_incorrect,
        }

    # Identify outlier models (those with lowest agreement rates)
    sorted_models = sorted(
        model_consistency.items(), key=lambda x: x[1]["confidence_weighted_agreement"]
    )

    return {
        "model_consistency": model_consistency,
        "debate_consensus": debate_consensus,
        "outlier_models": sorted_models[:3]
        if len(sorted_models) >= 3
        else sorted_models,
        "most_consistent_models": sorted_models[-3:][::-1]
        if len(sorted_models) >= 3
        else sorted_models[::-1],
    }


config = Config()
sample_debates_list = list(config.sample_debates_dir.glob("*.json"))

# Execute the analysis
results = analyze_judge_consistency(sample_debates_list)

# Print a summary report
print("Judge Model Consistency Analysis")
print("===============================\n")

print("Model Agreement Rates (Simple and Confidence-Weighted):")
for model, stats in sorted(
    results["model_consistency"].items(),
    key=lambda x: -x[1]["confidence_weighted_agreement"],
):
    print(f"  {model}:")
    print(
        f"    - Simple agreement rate: {stats['agreement_rate']:.2f} ({stats['total_votes']} votes, {stats['disagreements']} disagreements)"
    )
    print(
        f"    - Confidence-weighted agreement: {stats['confidence_weighted_agreement']:.2f}"
    )
    print(f"    - Avg confidence: {stats['avg_confidence']:.2f}")
    print(
        f"    - Avg confidence when correct: {stats['avg_confidence_when_correct']:.2f}"
    )
    print(
        f"    - Avg confidence when incorrect: {stats['avg_confidence_when_incorrect']:.2f}"
    )
    print(f"    - Calibration gap: {stats['calibration_gap']:.2f}")

print("\nOutlier Models (Most Inconsistent by Confidence-Weighted Score):")
for model, stats in results["outlier_models"]:
    print(
        f"  {model}: {stats['confidence_weighted_agreement']:.2f} confidence-weighted agreement"
    )

print("\nDebate Consensus Summary:")
for debate_id, data in results["debate_consensus"].items():
    simple_winner = data["simple_consensus"]
    weighted_winner = data["weighted_consensus"]
    prop_votes = data["vote_count"]["proposition"]
    opp_votes = data["vote_count"]["opposition"]
    prop_conf = data["confidence_weighted"]["proposition"]
    opp_conf = data["confidence_weighted"]["opposition"]
    print(f"  Debate {debate_id}:")
    print(f"    - Simple consensus: {simple_winner} wins ({prop_votes}-{opp_votes})")
    print(
        f"    - Weighted consensus: {weighted_winner} wins ({prop_conf:.2f}-{opp_conf:.2f})"
    )
