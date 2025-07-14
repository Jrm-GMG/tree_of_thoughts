"""
Tasks for Tree of Thoughts framework
"""

from tree_of_thoughts.base_task import Task
from .game24_task import Game24Task
from .ace_reason_math_task import AceReasonMathTask
from .gsm8k_task import GSM8KTask
from . import prompt

__all__ = [
    'Task',
    'Game24Task',
    'AceReasonMathTask',
    'GSM8KTask',
    'prompt'
]
