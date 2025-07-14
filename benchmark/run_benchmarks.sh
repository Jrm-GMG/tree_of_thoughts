#!/bin/bash
# Run several benchmarks sequentially
set -e

# Math reasoning benchmark
python3 -m benchmark math

# GSM8K benchmark (analytics mode)
python3 -m benchmark gsm8k

# Model comparison benchmarks
python3 -m benchmark model-comparison --task math

# GSM8K benchmark (A* only mode)
python3 -m benchmark gsm8k --mode astar

echo "All benchmarks completed."
