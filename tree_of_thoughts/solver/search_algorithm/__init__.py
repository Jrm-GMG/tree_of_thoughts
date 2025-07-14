"""Algorithms for Tree of Thoughts framework"""

from ..reasoning.generation import Generation
from ..reasoning.evaluation import Evaluation
from .base import SearchStrategy
from .bfs import BFSStrategy
from .dfs import DFSStrategy
from .astar import AStarStrategy

# Legacy alias for prompts module
from ..reasoning import prompt as prompts

__all__ = [
    'Generation',
    'Evaluation',
    'SearchStrategy',
    'BFSStrategy',
    'DFSStrategy',
    'AStarStrategy',
    'prompts',
]
