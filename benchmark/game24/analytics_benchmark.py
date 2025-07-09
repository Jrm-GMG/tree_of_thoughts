"""Analytics Game of 24 Benchmark with detailed comparison"""

import argparse
import os
import sys

# Ensure tree_of_thoughts is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
tot_root = os.path.dirname(os.path.dirname(current_dir))
if tot_root not in sys.path:
    sys.path.insert(0, tot_root)

from benchmark.utils.config.loader import load_config
from benchmark.utils.config.setup import initialize_llm, initialize_task, load_task_dataset
from benchmark.utils.execution.runners import ComparisonRunner
from benchmark.utils.analysis.metrics import ResultAnalyzer
from benchmark.utils.analysis import (
    build_comparator_from_results,
    SuccessOnlyTable,
)


def main():
    """Run the Game of 24 analytics benchmark."""
    parser = argparse.ArgumentParser(description="Game of 24 Analytics Benchmark")
    parser.add_argument(
        "--config", default="game24/analytics.yaml", help="Configuration file"
    )
    parser.add_argument("--num-puzzles", type=int, help="Override number of puzzles")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    if args.num_puzzles:
        config["num_puzzles"] = args.num_puzzles

    print("=" * 80)
    print("GAME OF 24 - ANALYTICS BENCHMARK")
    print("Detailed Strategy Comparison")
    print("=" * 80)

    # Initialize and run
    llm = initialize_llm(config)
    task = initialize_task(config)
    dataset = load_task_dataset(config)

    runner = ComparisonRunner(config, llm, task)
    results = runner.run(dataset)

    # Detailed analysis
    analyzer = ResultAnalyzer(results["strategies"])

    comparator = build_comparator_from_results(results["strategies"])
    table = SuccessOnlyTable().build(comparator)

    # Display comparison matrix
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON MATRIX")
    print("=" * 80)
    print(analyzer.create_comparison_matrix())

    print("\n" + "=" * 80)
    print("PAIRWISE WIN COUNTS (rows show strategy successes where columns failed)")
    print("=" * 80)
    print(table)

    # Pairwise comparisons
    if config.get("analytics", {}).get("show_pairwise_stats"):
        print("\n" + "=" * 80)
        print("PAIRWISE COMPARISONS")
        print("=" * 80)

        pairwise = analyzer.pairwise_comparison()
        for comparison, stats in pairwise.items():
            print(f"\n{comparison}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

    # Save results
    runner.save_results(results)

    # Export detailed analysis
    if "metrics" in config.get("output", {}):
        analyzer.export_analysis(config["output"]["metrics"])
        print(f"\nDetailed metrics saved to: {config['output']['metrics']}")


if __name__ == "__main__":
    main()
