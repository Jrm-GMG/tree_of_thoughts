"""Math solver using direct prompting without step-by-step reasoning."""

from baseline.base_solver import BaselineBaseSolver
from tasks.prompt import IO_PROMPT


class MathDirectSolver(BaselineBaseSolver):
    """Direct solver for math problems."""

    def get_prompt(self, problem_prompt: str, attempt: int) -> str:
        """Return the standardized IO prompt."""
        return IO_PROMPT.format(input=problem_prompt)


