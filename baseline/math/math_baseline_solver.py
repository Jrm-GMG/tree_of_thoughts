"""Math-specific baseline solver."""

from baseline.base_solver import BaselineBaseSolver
from tasks.prompt import COT_PROMPT


class MathBaselineSolver(BaselineBaseSolver):
    """Baseline solver for math problems using chain-of-thought prompting."""

    def get_prompt(self, problem_prompt: str, attempt: int) -> str:
        """Return the standardized CoT prompt."""
        return COT_PROMPT.format(input=problem_prompt)


