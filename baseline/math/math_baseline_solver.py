"""Math-specific baseline solver."""

from baseline.base_solver import BaselineBaseSolver
from tasks.prompts import COT_MATH_PROMPTS


class MathBaselineSolver(BaselineBaseSolver):
    """Baseline solver for math problems using direct prompting."""

    def get_prompt(self, problem_prompt: str, attempt: int) -> str:
        """Get prompt for math solving with different strategies for retries."""
        if attempt < len(COT_MATH_PROMPTS):            
            return COT_MATH_PROMPTS[attempt].format(problem=problem_prompt)
        return COT_MATH_PROMPTS[-1].format(problem=problem_prompt)


