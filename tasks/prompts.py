"""
Prompts for Tree of Thoughts tasks
"""

# ---------------------------------------------------------------------------
# Game of 24 prompts
# ---------------------------------------------------------------------------

def get_game24_task_prompt(numbers: list, rank: int, solve_rate: float) -> str:
    """Get the main task prompt for Game of 24"""
    return f"""You need to use the four numbers {numbers} and basic arithmetic operations (+, -, *, /) to reach the target number 24.

Problem Details:
- Numbers: {numbers}
- Difficulty Rank: {rank}
- Historical Solve Rate: {solve_rate:.1%}

Rules:
- Use each number exactly once
- You can use parentheses
- Only +, -, *, / operations are allowed
- The result must equal exactly 24

Think step by step about different ways to combine these numbers. When you find a solution, clearly state it using the format: <<<SOLUTION: your_expression>>>

For example: <<<SOLUTION: (1 + 2) * (3 + 4)>>>"""

GAME24_FORMAT_PROMPT = "Express your final answer as: <<<SOLUTION: arithmetic_expression>>> where the expression equals 24."

GAME24_SOLUTION_MARKER = "<<<SOLUTION:"

# Baseline and direct Game24 solver prompts
COT_GAME24_PROMPT = """Solve the Game of 24 using these numbers: {numbers}

Rules:
- Use each number exactly once
- Use only +, -, *, / operations
- You can use parentheses
- The result must equal exactly 24

Think step by step, then provide your final answer using this exact format:
<<<SOLUTION: your_arithmetic_expression>>>

Example: <<<SOLUTION: (1 + 2) * (3 + 4)>>>

Numbers to use: {numbers}

Solution:"""

IO_GAME24_PROMPT = """Use the numbers {numbers} with +, -, *, / exactly once each to make 24. Provide only the final expression in the format <<<SOLUTION: expression>>>"""

ENHANCED_COT_PROMPT = """You are solving the Game of 24. When you find a solution, use this exact format: <<<SOLUTION: your_expression>>>

For example: <<<SOLUTION: (1 + 2) * (3 + 4)>>>\n\nGenerate different mathematical approaches to combine the numbers step by step."""

# ---------------------------------------------------------------------------
# AceReason-Math prompts
# ---------------------------------------------------------------------------

def get_ace_math_task_prompt(problem: str) -> str:
    """Get the main task prompt for AceReason-Math"""
    return f"""Solve this mathematical problem step by step:

{problem}

Please show your reasoning process clearly, step by step. Work through the problem systematically and verify your calculations. When you reach your final answer, clearly state it in the format: \\boxed{{your_answer}}"""

ACE_MATH_FORMAT_PROMPT = "Express your final answer within \\boxed{} notation. For example: \\boxed{42} or \\boxed{x = 5}"

# Baseline and direct math solver prompts
COT_MATH_PROMPTS = [
    """Solve this mathematical problem step by step:

{problem}

Show your work clearly and provide your final answer in \\boxed{{}} notation.
For example: \\boxed{{42}} or \\boxed{{x = 5}}""",

    """I need you to solve this math problem carefully:

{problem}

Please:
1. Read the problem carefully
2. Identify what you need to find
3. Show all your work step by step
4. Put your FINAL answer in \\boxed{{answer}} format

Your final answer MUST be in \\boxed{{}} notation.""",

    """Let me try a different approach to solve this problem:

{problem}

Think about this systematically:
- What type of problem is this?
- What mathematical concepts apply?
- What's the most direct solution path?

Show your reasoning and put the final answer in \\boxed{{answer}} format.""",
]

IO_MATH_PROMPTS = [
    """Solve the following problem and give only the final answer in \\boxed{{answer}} format:\n\n{problem}""",
]

# ---------------------------------------------------------------------------
# GSM8K prompts
# ---------------------------------------------------------------------------

def get_gsm8k_task_prompt(problem: str) -> str:
    """Get the main task prompt for GSM8K"""
    return f"""Solve the following grade school math word problem step by step:\n\n{problem}\n\nEnd your answer with '#### <final_answer>'"""

GSM8K_FORMAT_PROMPT = "Provide the final numeric answer in the format: #### <answer>"

# Baseline and direct GSM8K solver prompts
COT_GSM8K_PROMPTS = [
    (
        """Solve the grade school math word problem below step by step. "
        "Explain each calculation and verify your reasoning. When you are "
        "confident in the solution, write the final numeric answer on its own "
        "line in the form '#### <answer>'.\n\n{problem}"""
    ),
]

IO_GSM8K_PROMPTS = [
    (
        """Solve the following problem and provide ONLY the numeric result. "
        "Return your final answer on a single line in the format '#### <answer>'."\n\n"{problem}"""
    ),
]

# ---------------------------------------------------------------------------
# Default prompts for Tree of Thoughts tasks
# ---------------------------------------------------------------------------

TOT_GENERATION_SYSTEM_PROMPTS = {
    'game24': (
        "You are assisting with a numeric puzzle that requires arithmetic reasoning. "
        "Suggest possible steps for combining the given numbers.",
    ),
    'math': (
        "You are assisting with mathematical reasoning. "
        "Generate step-by-step approaches with careful calculations.",
    ),
    'gsm8k': (
        "You are assisting with mathematical reasoning. "
        "Generate step-by-step approaches with careful calculations.",
    ),
}

TOT_EVALUATION_SYSTEM_PROMPTS = {
    'game24': (
        "You judge the potential of arithmetic reasoning steps toward reaching a numeric target.",
    ),
    'math': (
        "You judge the soundness of mathematical reasoning.",
    ),
    'gsm8k': (
        "You judge the soundness of mathematical reasoning.",
    ),
}

TOT_ROOT_NODE_TEMPLATES = {
    'game24': "Problem: Use numbers {numbers} to make 24",
}
