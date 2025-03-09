from config import Config
from topics.load_topics import load_topics
from ai_models.load_models import load_debate_models
from pathlib import Path
from scripts.count_debate_words import analyze_debate_token_counts
from scripts.continue_debate import continue_debate
from dotenv import load_dotenv
import random
from typing import List
import time

# Add a global failed debate queue
failed_debate_queue: List[Path] = []

def calculate_debate_costs(model_pair, token_stats, debate_models):
    prop_model, opp_model = model_pair

    # Get token rates for each model
    prop_rates = debate_models[prop_model]
    opp_rates = debate_models[opp_model]

    # Unpack mean token counts
    prompt_tokens_mean, _ = token_stats["prompt_tokens"]
    completion_tokens_mean, _ = token_stats["completion_tokens"]

    # Calculate costs for each model
    prop_cost = (prompt_tokens_mean * prop_rates[0] +
                 completion_tokens_mean * prop_rates[1]) / 10**6 # Convert to dollars
    opp_cost = (prompt_tokens_mean * opp_rates[0] +
                completion_tokens_mean * opp_rates[1]) / 10**6 # Convert to dollars

    return prop_cost, opp_cost

def process_failed_debates():
    global failed_debate_queue

    if not failed_debate_queue:
        print("\nNo failed debates to process.")
        return

    print(f"\nAttempting to process {len(failed_debate_queue)} failed debates...")

    retry_queue = []
    for debate_path in failed_debate_queue:
        print(f"\nRetrying debate at: {debate_path}")
        try:
            continue_debate(debate_path)
            print("Debate successfully completed on retry")
        except Exception as e:
            print(f"Debate failed again with error: {str(e)}")
            retry_queue.append(debate_path)

    failed_debate_queue = retry_queue

    if failed_debate_queue:
        print(f"\n{len(failed_debate_queue)} debates still failed after retry.")
    else:
        print("\nAll failed debates were successfully processed.")

def run_sample_debates() -> None:
    global failed_debate_queue
    config = Config()
    topics = load_topics(config)
    debate_models = load_debate_models(config)
    # Get token statistics
    token_stats = analyze_debate_token_counts(config)

    # Get all possible model pairs (no permutations, just combinations)
    model_names = list(debate_models.keys())
    model_pairs = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model_pairs.append((model_names[i], model_names[j]))

    num_pairs = max(1, round(len(model_pairs) * 0.025))

    print(f"Total possible pairs: {len(model_pairs)}")
    print(f"Number of pairs to run: {num_pairs} (will result in {num_pairs * 2} debates)")

    selected_pairs = random.sample(model_pairs, k=num_pairs)

    debates = []
    debate_counter = 1

    for model_a, model_b in selected_pairs:
        topic1 = random.choice(topics)
        topic2 = random.choice(topics)

        # First direction: A prop, B opp
        prop_cost1, opp_cost1 = calculate_debate_costs((model_a, model_b), token_stats, debate_models)
        debates.append({
            "topic": topic1,
            "prop": model_a,
            "opp": model_b,
            "output": config.sample_debates_dir / f"sample_debate_{debate_counter}.json",
            "prop_cost": prop_cost1,
            "opp_cost": opp_cost1
        })
        debate_counter += 1

        # Second direction: B prop, A opp
        prop_cost2, opp_cost2 = calculate_debate_costs((model_b, model_a), token_stats, debate_models)
        debates.append({
            "topic": topic2,
            "prop": model_b,
            "opp": model_a,
            "output": config.sample_debates_dir / f"sample_debate_{debate_counter}.json",
            "prop_cost": prop_cost2,
            "opp_cost": opp_cost2
        })
        debate_counter += 1

    # Print debate details
    for i, debate in enumerate(debates):
        print(f"\nDebate {i+1}:")
        print(f"Proposition: {debate['prop']}")
        print(f"Opposition: {debate['opp']}")
        print(f"Topic: {debate['topic']}")
        print("Estimated costs:")
        print(f"  Proposition model cost: ${debate['prop_cost']:.4f}")
        print(f"  Opposition model cost: ${debate['opp_cost']:.4f}")
        print(f"  Total debate cost: ${debate['prop_cost'] + debate['opp_cost']:.4f}")

    input("\nPress Enter to start running the debates...")

    for debate in debates:
        print(f"\nRunning debate on: {debate['topic']}")
        print(f"{debate['prop']} vs {debate['opp']}")
        debate_path:Path = debate["output"]
        debate_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            config.debate_service.run_debate(
                proposition_model=debate["prop"],
                opposition_model=debate["opp"],
                motion=debate["topic"],
                path_to_store=debate_path,
            )
        except Exception as e:
            print(f"Debate failed with error: {str(e)}")
            failed_debate_queue.append(debate_path)
            continue

    # After all debates are done, process the failed ones
    print("\nInitial debates completed. Waiting 60 seconds before processing failed debates...")
    time.sleep(60)  # Wait a minute before processing failed debates
    process_failed_debates()

if __name__ == "__main__":
    load_dotenv()
    run_sample_debates()
