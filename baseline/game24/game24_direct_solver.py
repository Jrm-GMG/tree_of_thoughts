"""Game24 solver using direct prompting without reasoning."""

from baseline.base_solver import BaselineBaseSolver
from tasks.prompt import IO_PROMPT


class Game24DirectSolver(BaselineBaseSolver):
    """Direct solver for Game of 24."""

    def get_prompt(self, problem_prompt: str, attempt: int) -> str:
        """Return the standardized IO prompt."""
        return IO_PROMPT.format(input=problem_prompt)


