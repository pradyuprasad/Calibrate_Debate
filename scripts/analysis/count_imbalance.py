import pandas as pd
import json
from .load_data import load_debate_data

def recommend_balanced_unique_debate_schedule(output_file="balance_debate.json"):
    """
    Creates a balanced debate schedule with unique model pairings and
    saves the recommendations to a JSON file.

    Args:
        output_file: Path to save the recommended debate pairings (default: balance_debate.json)

    Returns:
        List of recommended debate pairings
    """
    # Load the existing debate data
    data = load_debate_data()

    # Track each model's current debate counts
    model_roles = {}
    for model_name, stats in data.model_stats.items():
        prop_debates = stats.prop_wins + stats.prop_losses
        opp_debates = stats.opp_wins + stats.opp_losses

        model_roles[model_name] = {
            'prop_debates': prop_debates,
            'opp_debates': opp_debates,
            'total_debates': stats.debates
        }

    # Track existing debate pairings (which models have already debated each other)
    existing_pairings = set()
    for debate in data.debates:
        pair_key = (debate.proposition_model, debate.opposition_model)
        existing_pairings.add(pair_key)

    # Calculate the target number of debates per role
    # We'll use the max of current debates to ensure we only add debates
    max_prop = max(stats['prop_debates'] for stats in model_roles.values())
    max_opp = max(stats['opp_debates'] for stats in model_roles.values())

    # Calculate how many more debates each model needs in each role
    for model, stats in model_roles.items():
        stats['prop_needed'] = max_prop - stats['prop_debates']
        stats['opp_needed'] = max_opp - stats['opp_debates']


    # Display current state and needs
    print("\n=== Current Model Debate Participation ===\n")
    df = pd.DataFrame.from_dict(model_roles, orient='index')
    df.index.name = 'model'
    df.reset_index(inplace=True)
    print(df[['model', 'prop_debates', 'opp_debates', 'total_debates']].to_string(index=False))

    print("\n=== Debates Needed to Balance Participation ===\n")
    print(df[['model', 'prop_needed', 'opp_needed']].to_string(index=False))

    # Generate recommended debate pairings
    recommended_debates = []

    # Create a scoring function to prioritize model pairings
    def score_pairing(prop_model, opp_model):
        # Avoid same model debating itself
        if prop_model == opp_model:
            return -1000

        # Avoid repeated pairings
        if (prop_model, opp_model) in existing_pairings:
            return -500

        # Prioritize based on need
        prop_need = model_roles[prop_model]['prop_needed']
        opp_need = model_roles[opp_model]['opp_needed']

        # Higher score for pairings that satisfy higher needs
        return prop_need + opp_need

    # Create all possible pairings and score them
    models = list(model_roles.keys())
    all_pairings = []

    for prop_model in models:
        for opp_model in models:
            if prop_model != opp_model:  # Skip same model debating itself
                score = score_pairing(prop_model, opp_model)
                if score > 0:  # Only consider valid pairings
                    all_pairings.append({
                        'proposition': prop_model,
                        'opposition': opp_model,
                        'score': score
                    })

    # Sort pairings by score (highest first)
    all_pairings.sort(key=lambda x: x['score'], reverse=True)

    # Track which models we've paired in our new recommendations
    new_pairings = set()

    # Keep adding debates until we run out of valid pairings
    for pairing in all_pairings:
        prop_model = pairing['proposition']
        opp_model = pairing['opposition']

        # Skip if either model has no more needs
        if model_roles[prop_model]['prop_needed'] <= 0 or model_roles[opp_model]['opp_needed'] <= 0:
            continue

        # Skip if we've already paired these models in our new recommendations
        if (prop_model, opp_model) in new_pairings:
            continue

        # Add this debate to recommendations
        recommended_debates.append({
            'proposition': prop_model,
            'opposition': opp_model
        })

        # Update the tracking
        new_pairings.add((prop_model, opp_model))

        # Update the needs
        model_roles[prop_model]['prop_needed'] -= 1
        model_roles[opp_model]['opp_needed'] -= 1

    # Save the recommended debates to JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(recommended_debates, f, indent=2)
        print(f"\nRecommended debates saved to {output_file}")
    except Exception as e:
        print(f"\nError saving to {output_file}: {str(e)}")

    # Output the recommended debates
    print(f"\n=== Recommended Debates ({len(recommended_debates)}) ===\n")
    if recommended_debates:
        recommendations_df = pd.DataFrame(recommended_debates)
        print(recommendations_df.to_string(index=False))

        # Show what the balance would look like after these debates
        print("\n=== Projected Balance After Recommended Debates ===\n")
        projected = {model: stats.copy() for model, stats in model_roles.items()}

        for debate in recommended_debates:
            prop_model = debate['proposition']
            opp_model = debate['opposition']

            # Update projected counts
            projected[prop_model]['prop_debates'] += 1
            projected[opp_model]['opp_debates'] += 1
            projected[prop_model]['total_debates'] += 1
            projected[opp_model]['total_debates'] += 1

        # Convert to DataFrame for display
        projected_df = pd.DataFrame.from_dict(projected, orient='index')
        projected_df.index.name = 'model'
        projected_df.reset_index(inplace=True)

        # Calculate imbalance metrics
        projected_df['p_o_diff'] = projected_df['prop_debates'] - projected_df['opp_debates']
        mean_debates = projected_df['total_debates'].mean()
        projected_df['debate_dev'] = projected_df['total_debates'] - mean_debates

        print(projected_df[['model', 'prop_debates', 'opp_debates', 'p_o_diff', 'total_debates', 'debate_dev']].to_string(index=False))
    else:
        print("No additional debates needed for balance.")

    return recommended_debates

if __name__ == "__main__":
    recommend_balanced_unique_debate_schedule()
