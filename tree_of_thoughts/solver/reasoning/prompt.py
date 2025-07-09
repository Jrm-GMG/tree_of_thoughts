"""
Prompts for Tree of Thoughts components
"""

# Generation prompts
GENERATION_SYSTEM_PROMPTS = {
    'default': (
        "You are assisting with complex reasoning tasks. "
        "Generate possible next steps and solution strategies in a logical manner."
    ),
}

def get_generation_user_prompt(mode: str, context: str, num_to_generate: int) -> str:
    """Get user prompt for generation based on mode"""
    if mode == 'propose':
        return (
            f"Current path:\n{context}\n\n"
            f"Propose {num_to_generate} distinct next steps. Number them 1., 2., etc."
        )
    else:  # sample
        return f"Current path:\n{context}\n\nWhat would be a good next step?"

# Evaluation prompts
EVALUATION_SYSTEM_PROMPTS = {
    'default': (
        "You judge the potential of reasoning steps toward solving the given problem."
    ),
}

def get_evaluation_user_prompt(mode: str, context: str, target: str = "") -> str:
    """Get user prompt for single-path evaluation based on mode"""
    if mode == 'value':
        prompt = f"""Current solution path:
{context}

On a scale of 0-100, how promising is this solution path?
{f'Target: {target}' if target else ''}

Respond with 'grade{{value}}' where value is between 0 and 100."""
    else:  # vote
        prompt = f"""Current solution path:
{context}

Vote on whether this path should be continued or abandoned.
Respond with 'CONTINUE' or 'ABANDON' and a brief reason."""
    
    return prompt


def get_vote_prompt(instruction: str, choices: list) -> str:
    """Prompt for comparing multiple choices following the ToT paper"""
    formatted_choices = "\n".join(f"{i+1}. {choice}" for i, choice in enumerate(choices))
    return (
        f"{instruction}\n\n"
        "Given an instruction and several choices, decide which choice is most promising. "
        "Analyze each choice in detail, then conclude in the last line \"The best choice is {s}\", "
        "where s the integer id of the choice.\n\n"
        f"Choices:\n{formatted_choices}"
    )

# A* Search Strategy prompts
VALIDITY_EVALUATION_PROMPT = """Evaluate the validity of this reasoning chain.

Reasoning chain:
{reasoning_chain}

A reasoning step is VALID if it:
- Contains no logical errors
- Can be directly entailed from previous steps OR
- At least does not contradict previous steps
- Has correct calculations (for math/logic problems)

Respond with `grade{{value/4}}` where `value` is a number from 1 to 4:
1 = Contains major logical errors or contradictions
2 = Some logical issues but mostly correct
3 = Valid reasoning with minor issues
4 = Completely valid and logically sound

Answer:"""

COHERENCE_EVALUATION_PROMPT = """Evaluate the coherence of this reasoning chain.

Reasoning chain:
{reasoning_chain}

A reasoning step is COHERENT if:
- Its preconditions are satisfied by previous steps
- All values/concepts used are explained in prior steps
- The flow follows a logical sequence without gaps

Respond with `grade{{value/4}}` where `value` is a number from 1 to 4:
1 = Incoherent, major missing preconditions
2 = Some missing context but followable
3 = Mostly coherent with minor gaps
4 = Fully coherent with all preconditions met

Answer:"""