"""
Tree of Thoughts Framework

A framework for implementing Tree of Thoughts (ToT) reasoning with Large Language Models.
"""

# Import core components
from .tree_node import TreeNode
from .llm_instance import LLMInstance
from .solver.reasoning.enums import GenerationMode, EvaluationMode

# Import main modules
from .solver.reasoning.generation import Generation
from .solver.reasoning.evaluation import Evaluation

# Import strategies
from .solver.search_algorithm.bfs import BFSStrategy
from .solver.search_algorithm.dfs import DFSStrategy
from .solver.search_algorithm.astar import AStarStrategy
from .solver.search_algorithm.base import SearchStrategy

# Import solver components
from .solver.reasoning.tree_of_toughts import TreeOfThoughts

# Version info
__version__ = "1.0.0"

__all__ = [
    # Main class
    'TreeOfThoughts',
    
    # Core
    'TreeNode',
    'LLMInstance',
    'GenerationMode',
    'EvaluationMode',
    
    # Components
    'Generation',
    'Evaluation',
    'SearchStrategy',
    
    # Strategies
    'BFSStrategy',
    'DFSStrategy',
    'AStarStrategy',
    
]


