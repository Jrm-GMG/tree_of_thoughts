"""Model comparison utilities"""

import time
import pandas as pd
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

from tree_of_thoughts.llm_instance import LLMInstance
from .runners import ComparisonRunner
from ..analysis.metrics import ResultAnalyzer
from ..helpers import ensure_directory_exists

import gc


class ModelComparator:
    """Compares performance across different models"""

    def __init__(self, config: Dict[str, Any], task):
        """Initialize with configuration and the task to evaluate."""
        self.config = config
        self.task = task
        self.models = config.get("models", [])
        self.strategies = config.get("strategies", ["baseline", "bfs"])
        self.verbose = True  # TODO : put as arguments
        self.current_model = None

    def run_comparison(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Run comparison across all models"""
        results_by_model = {}

        for model_name in self.models:
            print(f"\n{'='*60}")
            print(f"Testing Model: {model_name}")
            print(f"{'='*60}")

            # Update config with current model
            model_config = self.config.copy()
            model_config["model"] = model_name

            # Get torch dtype
            dtype_str = self.config.get("torch_dtype", "float16")
            if dtype_str == "float16":
                torch_dtype = torch.float16
            elif dtype_str == "float32":
                torch_dtype = torch.float32
            elif dtype_str == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16  # default

            # Initialize LLM for this model
            llm = LLMInstance(
                model_name=model_name,
                device=self.config.get("device", "cuda"),
                torch_dtype=torch_dtype,
                load_in_8bit=self.config.get("load_in_8bit", False),
                load_in_4bit=self.config.get("load_in_4bit", False),
                max_length=self.config.get("max_length", 2048),
            )
            self.current_model = llm
            # Run comparison for this model
            runner = ComparisonRunner(model_config, llm, self.task)
            model_results = runner.run(dataset)

            results_by_model[model_name] = model_results
            self._cleanup_gpu_memory()
            time.sleep(1)

        # Analyze across models
        summary = self._create_model_summary(results_by_model)

        return {
            "models": results_by_model,
            "summary": summary,
            "best_model": self._find_best_model(results_by_model),
        }

    def _create_model_summary(self, results_by_model: Dict) -> pd.DataFrame:
        """Create summary comparing all models"""
        data = []

        for model, results in results_by_model.items():
            for strategy, metrics in results["strategies"].items():
                row = {
                    "Model": model,
                    "Strategy": strategy,
                    "Success Rate": metrics["success_rate"],
                    "Avg Time": metrics["avg_time"],
                    "Total Solved": metrics["total_solved"],
                }
                data.append(row)

        df = pd.DataFrame(data)
        pivot = df.pivot(index="Model", columns="Strategy", values="Success Rate")

        # Ensure consistent DataFrame output even if only one model or strategy
        if not isinstance(pivot, pd.DataFrame):
            pivot = pivot.to_frame()

        return pivot

    def _find_best_model(self, results_by_model: Dict) -> Dict[str, str]:
        """Find best model for each strategy"""
        best_by_strategy = {}

        for strategy in self.strategies:
            best_model = max(
                results_by_model.items(),
                key=lambda x: x[1]["strategies"][strategy]["success_rate"],
            )[0]
            best_by_strategy[strategy] = best_model

        return best_by_strategy

    def save_results(self, results: Dict[str, Any]):
        """Save comparison results"""
        output = self.config.get("output", {})

        if "summary" in output:
            ensure_directory_exists(output["summary"])
            results["summary"].to_csv(output["summary"])
            print(f"\nSummary saved to: {output['summary']}")

        if "detailed" in output:
            import json

            ensure_directory_exists(output["detailed"])
            with open(output["detailed"], "w") as f:
                json.dump(
                    {
                        "best_models": results["best_model"],
                        "model_count": len(self.models),
                        "strategy_count": len(self.strategies),
                    },
                    f,
                    indent=2,
                )

    def _cleanup_gpu_memory(self):
        """Free GPU memory after model testing"""
        if self.verbose:
            print("\nCleaning up GPU memory...")

        try:
            # Delete current model if exists
            if self.current_model:
                if hasattr(self.current_model, "model"):
                    del self.current_model.model
                if hasattr(self.current_model, "tokenizer"):
                    del self.current_model.tokenizer
                if hasattr(self.current_model, "pipeline"):
                    del self.current_model.pipeline
                del self.current_model
                self.current_model = None

            # Force garbage collection first
            gc.collect()
        except Exception as e:
            if self.verbose:
                print(f"  Warning during cleanup: {str(e)}")
