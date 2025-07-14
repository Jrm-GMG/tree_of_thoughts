"""GSM8K Benchmark (A* only)"""

import argparse
import os
import sys

from benchmark.utils.config.loader import load_config
from benchmark.utils.config.setup import initialize_llm, initialize_task, load_task_dataset
from benchmark.utils.execution.runners import ComparisonRunner

# Ensure tree_of_thoughts is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
tot_root = os.path.dirname(os.path.dirname(current_dir))
if tot_root not in sys.path:
    sys.path.insert(0, tot_root)


def main():
    """Run the GSM8K benchmark using only the A* strategy."""
    parser = argparse.ArgumentParser(description="GSM8K Benchmark")
    parser.add_argument(
        "--config", default="gsm8k/astar.yaml", help="Configuration file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 80)
    print("GSM8K REASONING - ASTAR BENCHMARK")
    print("=" * 80)

    llm = initialize_llm(config)
    task = initialize_task(config)
    dataset = load_task_dataset(config)

    runner = ComparisonRunner(config, llm, task)
    results = runner.run(dataset)

    # Display basic statistics for the A* run
    astar_results = results.get("strategies", {}).get("astar", {})
    if astar_results:
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(
            f"Success Rate: {astar_results['success_rate'] * 100:.1f}% "
            f"({astar_results['total_solved']}/{astar_results['total_problems']})"
        )
        print(f"Average Time: {astar_results['avg_time']:.2f}s")
        if "avg_nodes_explored" in astar_results:
            print(f"Average Nodes Explored: {astar_results['avg_nodes_explored']:.2f}")
        if "avg_max_depth" in astar_results:
            print(f"Average Max Depth: {astar_results['avg_max_depth']:.2f}")

    runner.save_results(results)


if __name__ == "__main__":
    main()
