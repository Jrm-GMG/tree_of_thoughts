"""GSM8K solver using direct prompting."""

from baseline.base_solver import BaselineBaseSolver
from tasks.prompt import IO_PROMPT


class GSM8KDirectSolver(BaselineBaseSolver):
    """Direct solver for GSM8K without chain-of-thought."""

    def get_prompt(self, problem_prompt: str, attempt: int) -> str:
        """Return the standardized IO prompt."""
        return IO_PROMPT.format(input=problem_prompt)


