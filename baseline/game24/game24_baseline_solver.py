"""Game24-specific baseline solver."""

from typing import List

from baseline.base_solver import BaselineBaseSolver
from tasks.prompts import COT_GAME24_PROMPT


class Game24BaselineSolver(BaselineBaseSolver):    
    """Baseline solver for Game of 24 using direct prompting."""

    def get_prompt(self, problem_prompt: List[int], attempt: int) -> str:
        """Return the standard Game24 prompt."""
        return COT_GAME24_PROMPT.format(numbers=problem_prompt)


