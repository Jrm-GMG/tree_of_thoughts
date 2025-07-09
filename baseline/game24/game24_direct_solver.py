"""Game24 solver using direct prompting without reasoning."""

from typing import List

from baseline.base_solver import BaselineBaseSolver
from tasks.prompts import IO_GAME24_PROMPT


class Game24DirectSolver(BaselineBaseSolver):
    """Direct solver for Game of 24."""

    def get_prompt(self, problem_prompt: List[int], attempt: int) -> str:
        """Return the direct prompt."""
        return IO_GAME24_PROMPT.format(numbers=problem_prompt)


