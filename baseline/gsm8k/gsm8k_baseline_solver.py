"""GSM8K-specific baseline solver."""

from baseline.base_solver import BaselineBaseSolver
from tasks.prompts import COT_GSM8K_PROMPTS


class GSM8KBaselineSolver(BaselineBaseSolver):
    """Baseline solver for GSM8K using chain-of-thought prompting."""

    def get_prompt(self, problem_prompt: str, attempt: int) -> str:
        """Return the prompt for the given attempt."""
        if attempt < len(COT_GSM8K_PROMPTS):
            return COT_GSM8K_PROMPTS[attempt].format(problem=problem_prompt)
        return COT_GSM8K_PROMPTS[-1].format(problem=problem_prompt)


