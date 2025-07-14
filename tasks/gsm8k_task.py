"""
GSM8K task implementation for Tree of Thoughts framework
"""

from typing import Any, Dict, Optional
import re
from datasets import load_dataset
import random

from tree_of_thoughts.base_task import Task
from tree_of_thoughts.tree_node import TreeNode
from .prompt import get_gsm8k_task_prompt, GSM8K_FORMAT_PROMPT


class GSM8KTask(Task):
    """Task class for the GSM8K dataset."""

    def __init__(self, split: str = "main", streaming: bool = False, debug: bool = False):
        """Load the dataset and prepare internal structures.

        Args:
            split: Dataset configuration name (e.g. ``"main"``).
            streaming: Whether to stream the dataset from disk.
            debug: If True, enable additional logging.
        """
        self.dataset_name = "gsm8k"
        self.dataset_config = split
        self.streaming = streaming
        self.debug = debug

        if streaming:
            self.dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split="test",
                streaming=True,
            )
            self.problems = None
            self.total_problems = len(
                load_dataset(
                    self.dataset_name,
                    self.dataset_config,
                    split="test",
                )
            )
        else:
            self.dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split="test",
            )
            self.problems = {i: self.dataset[i] for i in range(len(self.dataset))}
            self.total_problems = len(self.dataset)
        self.current_problem = None

    def load_problem(self, problem_id: int) -> Dict[str, Any]:
        """Load a single problem and set current_problem.

        Args:
            problem_id: Index of the problem to load.

        Returns:
            A dictionary with id, question and answer fields.
        """
        if problem_id < 0:
            raise ValueError("Problem ID must be non-negative")

        if self.streaming:
            for i, item in enumerate(self.dataset):
                if i == problem_id:
                    problem_data = item
                    break
        else:
            if problem_id >= self.total_problems:
                raise ValueError(f"Problem ID must be < {self.total_problems}")
            problem_data = self.problems[problem_id]

        self.current_problem = {
            "id": problem_id,
            "question": problem_data["question"],
            "answer": problem_data["answer"],
        }
        return self.current_problem

    def get_task_prompt(self) -> str:
        """Return the main prompt describing the current question.

        Returns:
            The formatted prompt describing the loaded problem.
        """
        if not self.current_problem:
            raise ValueError("No problem loaded")
        return get_gsm8k_task_prompt(self.current_problem["question"])

    def get_format_prompt(self) -> str:
        """Instruction for formatting the final answer.

        Returns:
            Prompt string that tells the model how to format answers.
        """
        return GSM8K_FORMAT_PROMPT

    def get_prompt_key(self) -> str:
        """Key for selecting generation/evaluation prompts.

        Returns:
            Identifier used to choose prompt templates.
        """
        return "math"

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Parse the final answer token following ####.

        Args:
            text: Text containing the answer token.

        Returns:
            The extracted answer string if present, otherwise None.
        """
        match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
        if match:
            return match.group(1)
        return None

    def _verify_answer(self, predicted: str, correct: str) -> bool:
        """Simple string match verification.

        Args:
            predicted: Model-predicted answer.
            correct: Ground-truth answer to compare against.

        Returns:
            True if the answers match exactly.
        """
        return predicted.strip() == correct.strip()

    def check_stopping_criteria(self, node: TreeNode) -> bool:
        """Return True if the node contains the correct final answer.

        Args:
            node: Node whose content should be inspected.

        Returns:
            True when a valid final answer is found.
        """
        content = node.content
        answer = self._extract_final_answer(content)

        if self.debug:
            print(f"\nDEBUG: Checking node (depth {node.depth}):")
            if answer is not None:
                print(f"  Extracted answer: {answer}")
                if self.current_problem:
                    correct = self.current_problem["answer"].split("####")[-1].strip()
                    print(f"  Correct answer: {correct}")
            else:
                print("  X No answer extracted")

        if answer is None:
            return False

        if self.current_problem and self._verify_answer(answer, self.current_problem["answer"].split("####")[-1].strip()):
            node._valid_answer = answer
            return True

        return False

    def parse_solution(self, node: TreeNode) -> Any:
        """Extract the verified final answer from a node.

        Args:
            node: Node returned by the search strategy.

        Returns:
            The final answer string if one was verified, otherwise None.
        """
        if hasattr(node, "_valid_answer"):
            return node._valid_answer

        if self.debug:
            print(f"\nDEBUG: Parsing solution from node (depth {node.depth})")

        answer = self._extract_final_answer(node.content)

        if answer and self.current_problem and self._verify_answer(answer, self.current_problem["answer"].split("####")[-1].strip()):
            if self.debug:
                correct = self.current_problem["answer"].split("####")[-1].strip()
                print(f"  Extracted answer: {answer}")
                print(f"  Correct answer: {correct}")
            return answer

        if self.debug:
            if answer is not None:
                correct = self.current_problem["answer"].split("####")[-1].strip() if self.current_problem else ""
                print(f"  Extracted answer: {answer}")
                if correct:
                    print(f"  Correct answer: {correct}")
            print("  Failed to parse solution")

        return None

    def get_total_problems(self) -> int:
        """Total number of available problems.

        Returns:
            Number of problems in the dataset.
        """
        return self.total_problems

    def get_problem_info(self, problem_id: int) -> Dict[str, Any]:
        """Return metadata for a specific problem.

        Args:
            problem_id: Dataset index of the problem.

        Returns:
            Dictionary containing preview information about the problem.
        """
        if self.streaming:
            raise NotImplementedError
        if problem_id >= self.total_problems:
            raise ValueError(f"Problem ID must be < {self.total_problems}")
        problem_data = self.problems[problem_id]
        return {
            "id": problem_id,
            "question_preview": problem_data["question"][:100],
            "answer": problem_data["answer"],
        }

