import json
from pathlib import Path


def load_round_results(round_num):
    """Load results from round_[num]_results.json"""
    path = Path(f"tournament/round_{round_num}_results.json")
    with open(path, "r") as f:
        return json.load(f)


def calculate_elo(winner_rating, loser_rating, k=64, win_margin=0.5):
    """
    Calculate updated Elo ratings
    win_margin: Proportion of judges that voted for winner (0.5-1.0)
    k: Base K-factor (64)
    """
    expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
    expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))

    # win_margin goes from 0.5 (3-3 split) to 1.0 (6-0 sweep)
    # This makes k_adjust go from minimum to full k
    k_adjust = k * (win_margin - 0.5) * 2

    winner_new = winner_rating + k_adjust * (1 - expected_winner)
    loser_new = loser_rating + k_adjust * (0 - expected_loser)

    return winner_new, loser_new


def process_tournament():
    # Initialize ratings at 1000
    ratings = {}

    # Process each round
    for round_num in [1, 2, 3]:
        results = load_round_results(round_num)

        for match in results:
            prop_model = match["prop_model"]
            opp_model = match["opp_model"]

            # Initialize any new models
            if prop_model not in ratings:
                ratings[prop_model] = 1200
            if opp_model not in ratings:
                ratings[opp_model] = 1200

            # Process match result
            if match["winner"] == "proposition":
                winner, loser = prop_model, opp_model
                margin = match["margin"]
            else:
                winner, loser = opp_model, prop_model
                margin = match["margin"]

            new_winner, new_loser = calculate_elo(
                ratings[winner], ratings[loser], k=64, win_margin=0.5 + margin / 2
            )

            # Update ratings
            ratings[winner] = new_winner
            ratings[loser] = new_loser

            print(f"\nAfter {winner} vs {loser}:")
            print(f"{winner}: {ratings[winner]:.0f}")
            print(f"{loser}: {ratings[loser]:.0f}")

    print("\nFinal Ratings:")
    for model, rating in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"{model}: {rating:.0f}")


if __name__ == "__main__":
    process_tournament()
