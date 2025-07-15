# Tree of Thoughts 

This is a side project exploring <a href="https://arxiv.org/abs/2305.10601">Tree of Thoughts</a><sup>1</sup> method. ToT enables LLMs to explore multiple reasoning paths through backtracking, evaluation, and systematic search. The A* search strategy is an experimental extension showing how classical AI search algorithms can be adapted with neural heuristics.

## Jobs done
- **Multiple Search Algorithms** – BFS and DFS from the original ToT paper, plus a custom A* implementation with LLM-based heuristics
- **LLM Agnostic** – Compatible with any Hugging Face model
- **Modular Architecture** – Pluggable components for tasks, generation, evaluation, and search strategies
- **Benchmarking Suite** – Comprehensive evaluation on Game of 24, AceReason-Math, and GSM8K datasets

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/tree-of-thoughts.git
cd tree-of-thoughts

# Install dependencies
pip install -r requirements.txt
```
The code was tested on an RTX 3060 (12 GB VRAM) using CUDA 12.6 and PyTorch 2.7.1.

## Quick Start
```python
from tree_of_thoughts import (
    LLMInstance, Generation, Evaluation, AStarStrategy,
    TreeOfThoughts, GenerationMode, EvaluationMode
)
from tasks import AceReasonMathTask

# Initialize components
llm = LLMInstance(model_name="meta-llama/Llama-3.2-3B-Instruct", device="cuda")
task = AceReasonMathTask()
generation = Generation(llm, mode=GenerationMode.SAMPLE, num_generations=3)
evaluation = Evaluation(llm, mode=EvaluationMode.VALUE)
search = AStarStrategy(depth_limit=4)

# Create solver and solve
solver = TreeOfThoughts(task, llm, generation, evaluation, search)
solution, tree = solver.solve(problem_id=0)
print(f"Solution: {solution}")
solver.print_solution_path()
```

## A* Search Strategy: Detailed Explanation

The A* search strategy is a custom extension beyond the original ToT paper, designed to combine the completeness of breadth-first search with the efficiency of heuristic-guided exploration.

### Core Algorithm

A* maintains a priority queue of nodes ordered by their f-score:
```
f(n) = g(n) + h(n)
```

Where:
- **g(n)**: Actual cost from start to node n.
- **h(n)**: Heuristic estimate of cost from n to goal.

The algorithm always expands the node with the lowest f-score, making it both optimal and complete under certain conditions.

### Implementation Details

#### 1. G-Score Calculation (Path Cost)

As inspiration, I used the following paper: <a href="https://arxiv.org/abs/2502.12289">Evaluating Step-by-step Reasoning Traces: A Survey</a><sup>2</sup> .
Since the TOT evaluation method already evaluates **utility**, I decided to evaluate the **validity** and **coherence** of the partial COT from node to root with an LLM as a judge 
(hence following the philosophy of the TOT paper by prompting the LLM to evaluate).
The g-score evaluates the quality of the reasoning chain from root to current node using two LLM-based metrics:

**Validity Assessment**
```python
def _evaluate_validity(self, reasoning_chain: str) -> float:
    """Score logical validity of reasoning steps"""
    prompt = VALIDITY_EVALUATION_PROMPT.format(reasoning_chain=reasoning_chain)
    # LLM grades on scale 1-4:
    # 1 = Contains major logical errors
    # 2 = Some logical issues but mostly correct  
    # 3 = Valid reasoning with minor issues
    # 4 = Completely valid and logically sound
```

**Coherence Assessment**
```python
def _evaluate_coherence(self, reasoning_chain: str) -> float:
    """Score sequential coherence of reasoning"""
    prompt = COHERENCE_EVALUATION_PROMPT.format(reasoning_chain=reasoning_chain)
    # LLM grades on scale 1-4:
    # 1 = Incoherent, major missing preconditions
    # 2 = Some missing context but followable
    # 3 = Mostly coherent with minor gaps
    # 4 = Fully coherent with all preconditions met
```

The combined g-score is:
```python
g_score = 100 - (w_validity * validity + w_coherence * coherence)
```

This inversion ensures that higher quality paths have lower costs, as required by A*.

#### 2. H-Score Calculation (Heuristic)

The heuristic uses the node's evaluation value from the standard ToT evaluation (not proven to be admissible; TODO: need to work on how we can guarantee it never overestimates the actual cost to reach the goal):
```python
def _get_h_score(self, node: TreeNode) -> float:
    if node.value is not None:
        return node.value / 10  # Scale to 0-10 range
    return 5.0  # Neutral estimate for unevaluated nodes
```

#### 3. F-Score Combination

The final f-score balances exploration vs exploitation:
```python
f_score = (1 - h_weight) * g_score + h_weight * h_score
```

Default `h_weight = 0.7` prioritizes promising nodes while still considering path quality.

### Key Optimizations

1. **Caching System**: G-scores are cached by `(content, depth)` to avoid re-evaluating identical reasoning chains:
```python
cache_key = f"{node.content}_{node.depth}"
if cache_key not in self.g_score_cache:
    # Compute and cache g-score
```

2. **Closed Set Tracking**: Prevents re-expansion of already explored nodes

3. **Depth Limiting**: Configurable maximum depth prevents infinite expansion


### Configuration

```yaml
search:
  astar:
    depth_limit: 4
    weights:
      validity: 0.5      # Weight for validity in g-score
      coherence: 0.5     # Weight for coherence in g-score
      h_weight: 0.7      # Balance between g and h scores
```

## Running Benchmarks

The framework includes comprehensive benchmarks with configurable strategies:

```bash
# Game of 24 benchmark
python -m benchmark.game24.analytics_benchmark

# Math reasoning benchmark  
python -m benchmark.math.analytics_benchmark

# Model comparison across strategies
python -m benchmark.model_comparison.game24_comparison
```
## Or run them several benchmark in one go
bash benchmark/run_benchmarks.sh

Now you can go grab a coffee (or two ...)


### Benchmark Configuration

Benchmarks use YAML configuration files that support inheritance:

```yaml
extends: defaults.yaml

task: math
num_problems: 100

dataset:
  source: "huggingface"
  streaming: false

strategies:
  - baseline_cot
  - bfs
  - dfs
  - astar

output:
  results: "benchmark/results/math_analytics.csv"
```
## Result

This benchmark evaluates the TOT approach using the meta-llama/Llama-3.2-3B-Instruct model on the NVIDIA AceReason-Math dataset. We compare TOT performance on 100 randomly selected math problems (using seed 42) across different tree exploration strategies: A*, breadth-first search (BFS), depth-first search (DFS), and an oracle setup for CoT where the best of k = 10 samples is selected.

The TOT parameters are configured as follows: maximum depth of 5, 3 generations per node, and generation mode set to SAMPLE (see benchmark/config/math/analytics.yaml config file for more details). 
To reproduce these experiments, run `python3 -m benchmark math`.

### Strategy Performance Summary

| Strategy | Success Rate | Avg Time (s) | Total Solved | Total Problems | Avg Nodes |
|----------|-------------|--------------|--------------|----------------|-----------|
| **baseline_cot** (best of 10) | 0.36 | 132.65 | 36 | 100 | - |
| **bfs** | 0.90 | 69.97 | 90 | 100 | 18.68 |
| **dfs** | 0.87 | 233.32 | 87 | 100 | 54.15 |
| **A \*** | 0.99 | 249.72 | 99 | 100 | 48.25 |

### Pairwise Win Counts
*Rows show strategy successes where columns failed*

| Strategy A / Strategy B | **A \*** | **baseline_cot** (best of 10) | **bfs** | **dfs** |
|------------------------|-------|--------------|-----|-----|
| **A \*** | 0 | 63 | 9 | 12 |
| **baseline_cot** (best of 10) | 0 | 0 | 2 | 2 |
| **bfs** | 0 | 56 | 0 | 9 |
| **dfs** | 0 | 53 | 6 | 0 |

### Key Results
- **A \*** achieves the highest success rate at 99%
- **BFS** provides the fastest average solving time at ~70 seconds (baseline exlcude)
- **Baseline CoT** (best of 10) has the lowest success rate at 36%
- **DFS** uses the most nodes on average (54.15) but maintains good success rate (87%)

  
More experiments are on the way. The `model-comparison` mode will test `microsoft/Phi-4-mini-reasoning`, `meta-llama/Llama-3.2-3B-Instruct` and `Qwen/Qwen3-4B` across baseline prompting and ToT using BFS, DFS and A* on both the NVIDIA AceReason-Math and GSM8K datasets.

## Repository Structure
```
tree-of-thoughts/
│
├── tree_of_thoughts/              # Core framework
│   ├── llm_instance.py            # LLM wrapper
│   ├── tree_node.py               # Tree data structure
│   ├── base_task.py               # Task interface
│   └── solver/                   
│       ├── reasoning/             # Generation & evaluation modules
│       └── search_algorithm/      # BFS, DFS, A* strategies
│
├── tasks/                        # Task implementations
│   ├── game24_task.py            # Game of 24
│   ├── ace_reason_math_task.py   # Math reasoning = NVIDIA AceReason-Math
│   └── gsm8k_task.py             # GSM8K
│
├── baseline/                     # Baseline solvers (IO/CoT)
│   ├── game24/                  
│   ├── math/                    
│   └── gsm8k/                   
│
├── benchmark/                    # Evaluation suite
│   ├── config/                   # YAML configurations
│   ├── utils/                    # Runners, analysis tools
│   ├── game24/                   # game24 benchmarks
│   ├── math/                     # NVIDIA AceReason-Math benchmarks
│   ├── gsm8k/                    # Task-specific benchmarks
│   └── model_comparison/         # Cross-model evaluation
│
├── data/                         # Datasets
│   └── 24/24.csv                 # GSM8K
│
├── setup.py                     
├── requirements.txt             
└── README.md
```

## Sources

@misc{yao2023treethoughtsdeliberateproblem,

      title={Tree of Thoughts: Deliberate Problem Solving with Large Language Models}, 

      author={Shunyu Yao and Dian Yu and Jeffrey Zhao and Izhak Shafran and Thomas L. Griffiths and Yuan Cao and Karthik Narasimhan},

      year={2023},

      eprint={2305.10601},

      archivePrefix={arXiv},

      primaryClass={cs.CL},

      url={https://arxiv.org/abs/2305.10601}, 
}

@misc{lee2025evaluatingstepbystepreasoningtraces,

      title={Evaluating Step-by-step Reasoning Traces: A Survey}, 

      author={Jinu Lee and Julia Hockenmaier},

      year={2025},

      eprint={2502.12289},

      archivePrefix={arXiv},

      primaryClass={cs.CL},

      url={https://arxiv.org/abs/2502.12289}, 
}


