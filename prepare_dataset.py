import argparse
import json
import yaml
from datasets import load_dataset
from tqdm import tqdm
from routellm.controller import Controller
from routellm.routers.routers import ROUTER_CLS
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def create_rationale_dataset(battles_dataset, routers, config):
    # Load the full RouteLLM dataset
    battles_df = load_dataset(battles_dataset, split="train").to_pandas()

    # Initialize the RouteLLM controller
    controller = Controller(
        routers=routers,
        strong_model="GPT-4",  # Replace with your actual strong model identifier
        weak_model="LLaMA",    # Replace with your actual weak model identifier
        config=yaml.safe_load(open(config, "r")) if config else None,
        progress_bar=True,
    )

    # Specify the router and threshold to use
    router_name = routers[0]  # Use the first router in the list
    threshold = 0.5  # Set your desired threshold value

    # Create the rationale-augmented dataset
    rationale_dataset = []
    for _, row in tqdm(battles_df.iterrows(), total=len(battles_df)):
        # Extract the conversation or prompt
        prompt_data = json.loads(row['prompt'])  # Adjust based on actual data structure
        # Extract the user's message (assuming it's the last message)
        # print(prompt_data)
        prompt = prompt_data[-1]

        # Use the controller to route the prompt and get the rationale
        routed_model, rationale = controller.route(prompt, router_name, threshold)

        # Determine the label based on the routed model
        if routed_model == controller.model_pair.strong:
            label = 'strong'
        else:
            label = 'weak'

        rationale_dataset.append({"query": prompt, "label": label, "rationale": rationale})

    return rationale_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles_dataset", type=str, default="lmsys/lmsys-arena-human-preference-55k")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--routers", nargs="+", type=str, default=["keyword"], choices=list(ROUTER_CLS.keys()))
    args = parser.parse_args()

    # Create the rationale dataset
    rationale_dataset = create_rationale_dataset(args.battles_dataset, args.routers, args.config)

    # Save the rationale-augmented dataset
    with open("rationale_dataset.json", "w") as f:
        json.dump(rationale_dataset, f)

    print("Rationale dataset saved successfully.")
