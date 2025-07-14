""""
Abstract base class for Tree of Thoughts tasks
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from tree_of_thoughts.tree_node import TreeNode


class Task(ABC):
    """Abstract base class for specific tasks"""
    
    @abstractmethod
    def get_task_prompt(self) -> str:
        """Return the prompt explaining the task"""
        pass
    
    @abstractmethod
    def get_format_prompt(self) -> str:
        """Return prompt for formatting the answer"""
        pass

    @abstractmethod
    def get_prompt_key(self) -> str:
        """Return prompt key used for generation/evaluation"""
        pass
    
    @abstractmethod
    def check_stopping_criteria(self, node: TreeNode) -> bool:
        """Check if we should stop - found solution or reached target"""
        pass
    
    @abstractmethod
    def parse_solution(self, node: TreeNode) -> Any:
        """Parse the final solution from a node"""
        pass
    
    @abstractmethod
    def load_problem(self, problem_id: Any) -> Dict[str, Any]:
        """Load a specific problem instance"""
        pass