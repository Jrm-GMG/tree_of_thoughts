"""
Prompts for Tree of Thoughts solver
"""

# Root node content templates
ROOT_NODE_TEMPLATES = {
    'default': "Problem: {problem}",
    'general': "Starting to solve: {task_prompt}"
}

def get_root_node_content(problem_data, problem_type='default'):
    """Format the root node content based on problem type"""
    if problem_type == 'general':
        return ROOT_NODE_TEMPLATES['general'].format(task_prompt=problem_data)
    else:
        return ROOT_NODE_TEMPLATES['default'].format(problem=problem_data)
