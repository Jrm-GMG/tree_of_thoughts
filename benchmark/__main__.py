"""Main entry point for running benchmarks"""

import sys
import os
import argparse

# Add tree_of_thoughts to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
for path in (parent_dir, root_dir):
    if path not in sys.path:
        sys.path.insert(0, path)


def main():
    """Dispatch to the chosen benchmark based on CLI arguments."""
    parser = argparse.ArgumentParser(description="Tree of Thoughts Benchmark Runner")
    parser.add_argument(
        "benchmark",
        choices=["game24", "math", "gsm8k", "comprehensive", "model-comparison"],
        help="Benchmark type to run",
    )
    parser.add_argument(
        "--task", choices=["game24", "math"], help="Task for model comparison"
    )
    parser.add_argument(
        "--mode",
        choices=["analytics", "astar"],
        default="analytics",
        help="Benchmark mode for gsm8k",
    )

    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_args

    if args.benchmark == "game24":
        from benchmark.game24.analytics_benchmark import main
        main()
    elif args.benchmark == "math":
        from benchmark.math.analytics_benchmark import main
        main()
    elif args.benchmark == "gsm8k":
        if args.mode == "astar":
            from benchmark.gsm8k.astar_benchmark import main
        else:
            from benchmark.gsm8k.analytics_benchmark import main
        main()
    elif args.benchmark == "model-comparison":
        if args.task == "math":
            from benchmark.model_comparison.math_comparison import main
        else:
            from benchmark.model_comparison.game24_comparison import main
        main()


if __name__ == "__main__":
    main()
