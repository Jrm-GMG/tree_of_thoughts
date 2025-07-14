"""Base baseline solver for Tree of Thoughts framework."""

import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any

from tree_of_thoughts.base_task import Task
from tree_of_thoughts.tree_node import TreeNode


class BaselineBaseSolver(ABC):
    """Abstract base class for baseline solvers using direct prompting."""

    def __init__(self, llm, task: Task, temperature: float = 1.0):
        """Store references to the shared LLM instance, task and temperature."""
        self.llm = llm
        self.task = task
        # default temperature is 1.0 but can be overridden when instantiating
        self.temperature = temperature
    
    def solve(
        self,
        problem_id: Any,
        best_of: int = 1,
    ) -> Tuple[Optional[str], float, int]:
        """Solve the problem using direct prompting.

        When best_of is greater than 1 the LLM is asked to generate that
        many completions in a single call (oracle mode). The first valid
        response is returned.

        Args:
            problem_id: Identifier of the problem to load with the task.
            best_of: Number of completions to request from the LLM.

        Returns:
            Tuple (solution, time_taken, attempts) where solution is the
            best found answer or None.
        """

        overall_start = time.time()

        # Load problem using the task
        self.task.load_problem(problem_id)
        prompt_input = self.task.get_task_prompt()

        solution, elapsed, attempts = self._run(prompt_input, best_of)

        if solution is not None:
            return solution, elapsed, attempts

        return None, time.time() - overall_start, attempts

    def _run(
        self, problem_prompt: str, num_sequences: int
    ) -> Tuple[Optional[str], float, int]:
        """Run the LLM once and check all returned sequences."""

        start_time = time.time()
        attempt = 0
        prompt = self.get_prompt(problem_prompt, attempt)

        try:
            responses = self.llm.generate(
                prompt,
                max_new_tokens=1024,
                temperature=self.temperature,
                num_return_sequences=num_sequences,
            )

            if not isinstance(responses, list):
                responses = [responses]
            for resp in responses:
                response = resp if isinstance(resp, str) else str(resp)
                node = TreeNode(content=response, depth=1)
                if self.task.check_stopping_criteria(node):
                    solution = self.task.parse_solution(node)
                    if solution is not None:
                        return solution, time.time() - start_time, num_sequences

        except Exception as e:
            print(f"Error in baseline solver: {e}")

        return None, time.time() - start_time, num_sequences
    
    @abstractmethod
    def get_prompt(self, problem_prompt: Any, attempt: int) -> str:
        """Return the prompt for the model.

        Args:
            problem_prompt: Text of the task-specific prompt.
            attempt: The current attempt index starting at 0.

        Returns:
            The formatted prompt string to send to the LLM.
        """
        pass





