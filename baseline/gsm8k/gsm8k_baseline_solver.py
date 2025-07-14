"""GSM8K-specific baseline solver."""

from baseline.base_solver import BaselineBaseSolver
from tasks.prompt import COT_PROMPT


class GSM8KBaselineSolver(BaselineBaseSolver):
    """Baseline solver for GSM8K using chain-of-thought prompting."""

    def get_prompt(self, problem_prompt: str, attempt: int) -> str:
        """Return the standardized CoT prompt."""
        return COT_PROMPT.format(input=problem_prompt)


