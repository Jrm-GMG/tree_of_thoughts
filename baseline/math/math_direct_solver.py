"""Math solver using direct prompting without step-by-step reasoning."""

from baseline.base_solver import BaselineBaseSolver
from tasks.prompts import IO_MATH_PROMPTS


class MathDirectSolver(BaselineBaseSolver):
    """Direct solver for math problems."""

    def get_prompt(self, problem_prompt: str, attempt: int) -> str:
        """Return the single-shot prompt for the problem.

        Args:
            problem: The math question to be answered.
            attempt: Attempt index, starting at 0.

        Returns:
            Formatted prompt string for the LLM.
        """
        return IO_MATH_PROMPTS[0].format(problem=problem_prompt)


