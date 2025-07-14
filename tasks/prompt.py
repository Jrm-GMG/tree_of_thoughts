"""Prompts for Tree of Thoughts tasks.

This module provides succinct task descriptions and answer format
instructions used across baseline prompting (IO/COT) and the Tree of
Thoughts (ToT) reasoning methods.
"""


# ANSWER FORMATS

# how the final answer should be written for each task
# These tokens indicate the expected answer format without extra explanation
GAME24_FORMAT_PROMPT = (
    "Use the format <<<SOLUTION: expression>>> where `expression` is your final solution."
)
ACE_MATH_FORMAT_PROMPT = (
    "Write the final answer as \\boxed{value} with `value` being the result."
)
GSM8K_FORMAT_PROMPT = (
    "End with `####` followed by the numeric result, for example: #### 42."
)

# token that precedes a Game24 solution expression
GAME24_SOLUTION_MARKER = "<<<SOLUTION:"

# Direct answer (IO) prompting
IO_PROMPT = (
    "Answer the following question: {input}\n"
    "Answer:"
)

# Chain-of-Thought (CoT) prompting
COT_PROMPT = (
    "Answer the following question: {input}\n"
    "Think, step by step. Your output should include reasoning followed by the final answer.\n"
    "Solution:"
)

def get_game24_task_prompt(numbers: list) -> str:
    """Problem statement for the Game of 24 with format reminder."""

    return (
        f"Use the four numbers {numbers} with operations +, -, *, / to make 24. "
        f"Use each number exactly once. {GAME24_FORMAT_PROMPT}"
    )


def get_ace_math_task_prompt(problem: str) -> str:
    """Problem statement for AceReason-Math with format reminder."""

    return (
        f"Solve the following mathematical problem:\n\n{problem}\n\n"
        f"{ACE_MATH_FORMAT_PROMPT}"
    )


def get_gsm8k_task_prompt(problem: str) -> str:
    """Problem statement for GSM8K with format reminder."""

    return (
        f"Solve the following grade school math word problem:\n\n{problem}\n\n"
        f"{GSM8K_FORMAT_PROMPT}"
    )
