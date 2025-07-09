"""Analytics Math Benchmark with detailed metrics"""

import argparse
import os
import sys

from benchmark.utils.config.loader import load_config
from benchmark.utils.config.setup import (
    initialize_llm,
    initialize_task,
    load_task_dataset,
)
from benchmark.utils.execution.runners import ComparisonRunner
from benchmark.utils.analysis.metrics import ResultAnalyzer
from benchmark.utils.analysis import (
    build_comparator_from_results,
    SuccessOnlyTable,
)


# Ensure tree_of_thoughts is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
tot_root = os.path.dirname(os.path.dirname(current_dir))
if tot_root not in sys.path:
    sys.path.insert(0, tot_root)


def main():
    """Run the math analytics benchmark and print detailed metrics."""
    parser = argparse.ArgumentParser(description="Math Reasoning Analytics Benchmark")
    parser.add_argument(
        "--config", default="math/analytics.yaml", help="Configuration file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 80)
    print("MATH REASONING - ANALYTICS BENCHMARK")
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
    print("PERFORMANCE METRICS")
    print("=" * 80)
    print(analyzer.create_comparison_matrix())

    print("\n" + "=" * 80)
    print("PAIRWISE WIN COUNTS (rows show strategy successes where columns failed)")
    print("=" * 80)
    print(table)

    # Save all results
    runner.save_results(results)

    if "metrics" in config.get("output", {}):
        analyzer.export_analysis(config["output"]["metrics"])


if __name__ == "__main__":
    main()
