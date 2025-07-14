"""Common initialization utilities"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import torch
import random
from datasets import load_dataset

from tree_of_thoughts.llm_instance import LLMInstance
from tasks.game24_task import Game24Task
from tasks.ace_reason_math_task import AceReasonMathTask
from tasks.gsm8k_task import GSM8KTask
from tree_of_thoughts.solver.reasoning.generation import Generation
from tree_of_thoughts.solver.reasoning.evaluation import Evaluation
from tree_of_thoughts.solver.reasoning.enums import GenerationMode, EvaluationMode


def initialize_llm(config: Dict[str, Any]) -> LLMInstance:
    """Initialize LLM from configuration"""
    # Set seed for reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    random.seed(seed)

    # Get torch dtype
    dtype_str = config.get("torch_dtype", "float16")
    if dtype_str == "float16":
        torch_dtype = torch.float16
    elif dtype_str == "float32":
        torch_dtype = torch.float32
    elif dtype_str == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16  

    return LLMInstance(
        model_name=config["model"],
        device=config.get("device", "cuda"),
        torch_dtype=torch_dtype,
        load_in_8bit=config.get("load_in_8bit", False),
        load_in_4bit=config.get("load_in_4bit", False),
        max_length=config.get("max_length", 2048),
    )


def initialize_task(config: Dict[str, Any]) -> Any:
    """Initialize task based on configuration"""
    task_type = config.get("task")
    dataset_config = config.get("dataset", {})

    if task_type == "game24":
        # Get CSV path from config or use default
        csv_path = dataset_config.get("path", "data/24/24.csv")
        task = Game24Task(csv_path=csv_path, debug=config.get("debug", False))

        # difficulty filter if specified
        if config.get("hard_only", False):
            task.filter_by_difficulty(hard_only=True)

        return task

    elif task_type == "math":
        streaming = dataset_config.get("streaming", False)
        debug = config.get("debug", False)

        if dataset_config.get("source") == "file":
            raise NotImplementedError(
                "Math task requires a HuggingFace dataset; file sources are unsupported."
            )


        return AceReasonMathTask(streaming=streaming, debug=debug)

    elif task_type == "gsm8k":
        streaming = dataset_config.get("streaming", False)
        debug = config.get("debug", False)
        split = dataset_config.get("split", "main")
        if dataset_config.get("source") == "file":
            raise NotImplementedError(
                "GSM8K task requires a HuggingFace dataset; file sources are unsupported."
            )
        return GSM8KTask(split=split, streaming=streaming, debug=debug)

    else:
        raise ValueError(f"Unknown task type: {task_type}")


def load_task_dataset(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Load dataset metadata based on task configuration.

    Since tasks handle their own data loading, this returns metadata
    about what problems to use rather than the actual dataset.
    """
    task_type = config.get("task")

    if task_type == "game24":
        # Return metadata for Game24
        metadata = {
            "source": "csv",
            "num_puzzles": config.get("num_puzzles", 5),
            "hard_only": config.get("hard_only", False),
            "problem_ids": _get_game24_problem_ids(config),
        }
        return metadata

    elif task_type == "math":
        # Return metadata for Math
        metadata = {
            "source": "huggingface",
            "num_problems": config.get("num_problems", 10),
            "problem_ids": _get_math_problem_ids(config),
        }
        return metadata

    elif task_type == "gsm8k":
        metadata = {
            "source": "huggingface",
            "num_problems": config.get("num_problems", 10),
            "problem_ids": _get_gsm8k_problem_ids(config),
        }
        return metadata

    else:
        raise ValueError(f"Unknown task type: {task_type}")


def _select_problem_ids(total: int, requested: int, random_selection: bool) -> List[int]:
    """Return a list of problem IDs either sequentially or randomly."""
    requested = min(requested, total)
    if random_selection:
        return random.sample(range(total), requested)
    return list(range(requested))


def _get_math_problem_ids(config: Dict[str, Any]) -> List[int]:
    """Get problem IDs for the AceReason-Math dataset."""
    dataset_config = config.get("dataset", {})

    if "problem_ids" in dataset_config:
        return dataset_config["problem_ids"]

    num_problems = config.get("num_problems", 10)
    random_sel = dataset_config.get("random_selection", False)

    if random_sel:
        total_problems = len(load_dataset("nvidia/AceReason-Math", split="train"))
    else:
        total_problems = num_problems

    return _select_problem_ids(total_problems, num_problems, random_sel)


def _get_gsm8k_problem_ids(config: Dict[str, Any]) -> List[int]:
    """Get problem IDs for the GSM8K dataset."""
    dataset_config = config.get("dataset", {})

    if "problem_ids" in dataset_config:
        return dataset_config["problem_ids"]

    num_problems = config.get("num_problems", 10)
    random_sel = dataset_config.get("random_selection", False)

    if random_sel:
        split = dataset_config.get("split", "main")
        total_problems = len(
            load_dataset("gsm8k", name=split, split="test")
        )
    else:
        total_problems = num_problems

    return _select_problem_ids(total_problems, num_problems, random_sel)


def _get_game24_problem_ids(config: Dict[str, Any]) -> List[int]:
    """Get problem IDs for the Game24 dataset based on configuration."""
    dataset_config = config.get("dataset", {})

    # explicit specification of problem IDs
    if "problem_ids" in dataset_config:
        return dataset_config["problem_ids"]

    num_puzzles = config.get("num_puzzles", 5)
    random_sel = dataset_config.get("random_selection", False)

    if random_sel:
        path = dataset_config.get("path", "data/24/24.csv")
        df = pd.read_csv(path)
        total_puzzles = len(df)
    else:
        total_puzzles = num_puzzles

    return _select_problem_ids(total_puzzles, num_puzzles, random_sel)


def create_tot_components(llm: LLMInstance, task: Any, config: Dict[str, Any]) -> tuple:
    """Create ToT components (generation and evaluation)"""
    tot_config = config.get("tot", {})

    # Ask task for the appropriate prompt key
    if hasattr(task, "get_prompt_key"):
        prompt_key = task.get_prompt_key()
    else:
        prompt_key = "default"

    generation = Generation(
        llm=llm,
        mode=GenerationMode.SAMPLE,
        num_generations=tot_config.get("num_generations", 3),
        temperature=tot_config.get("temperature", 0.7),
        max_new_tokens=tot_config.get("max_new_tokens", 200),
        prompt_key=prompt_key,
    )

    evaluation = Evaluation(
        llm=llm,
        mode=EvaluationMode.VALUE,
        temperature=0.1,
        max_new_tokens=50,
        prompt_key=prompt_key,
    )

    return generation, evaluation
