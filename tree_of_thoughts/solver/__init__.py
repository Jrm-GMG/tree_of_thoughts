"""
Tree of Thoughts solver
"""

from .reasoning.tree_of_toughts import TreeOfThoughts
from . import prompts


__all__ = [
    'TreeOfThoughts',
    'prompts'
]

