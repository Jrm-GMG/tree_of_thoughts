"""GSM8K solver using direct prompting."""

from baseline.base_solver import BaselineBaseSolver
from tasks.prompts import IO_GSM8K_PROMPTS


class GSM8KDirectSolver(BaselineBaseSolver):
    """Direct solver for GSM8K without chain-of-thought."""

    def get_prompt(self, problem_prompt: str, attempt: int) -> str:
        """Return the direct prompt."""
        return IO_GSM8K_PROMPTS[0].format(problem=problem_prompt)


