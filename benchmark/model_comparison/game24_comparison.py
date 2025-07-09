"""Game of 24 Model Comparison Demo"""

import argparse
import os
import sys

# Ensure tree_of_thoughts is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
tot_root = os.path.dirname(os.path.dirname(current_dir))
if tot_root not in sys.path:
    sys.path.insert(0, tot_root)

from benchmark.utils.config.loader import load_config
from benchmark.utils.config.setup import initialize_task, load_task_dataset
from benchmark.utils.execution.model_comparator import ModelComparator


def main():
    """Execute the Game of 24 model comparison demo."""
    parser = argparse.ArgumentParser(description="Game of 24 Model Comparison")
    parser.add_argument(
        "--config", default="model_comparison/game24.yaml", help="Configuration file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 80)
    print("GAME OF 24 - MODEL COMPARISON")
    print("=" * 80)

    task = initialize_task(config)
    dataset = load_task_dataset(config)

    comparator = ModelComparator(config, task)
    results = comparator.run_comparison(dataset)

    # Display results
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    print(results["summary"])

    # Save results
    comparator.save_results(results)


if __name__ == "__main__":
    main()
