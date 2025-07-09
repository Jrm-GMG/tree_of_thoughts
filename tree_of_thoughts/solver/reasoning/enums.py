"""
Enumeration classes for Tree of Thoughts framework
"""

from enum import Enum


class GenerationMode(Enum):
    """Defines how the LLM will generate new thoughts"""
    SAMPLE = "sample"  # Generate independent samples
    PROPOSE = "propose"  # Generate multiple proposals in one go


class EvaluationMode(Enum):
    """Defines how nodes in the tree are evaluated"""
    VALUE = "value"  # Assign numeric scores (0-100)
    VOTE = "vote"  # Binary decision (continue/abandon)


class SearchStrategy(Enum):
    """Defines the tree exploration strategy"""
    BFS = "bfs"  # Breadth-first search 
    DFS = "dfs"  # Depth-first search 
    ASTAR = "astar"  # A* search 
