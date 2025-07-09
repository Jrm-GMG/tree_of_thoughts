"""Base search strategy classes"""
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Optional

from tree_of_thoughts.tree_node import TreeNode
from tree_of_thoughts.llm_instance import LLMInstance


class SearchStrategy(ABC):
    """Abstract base class for search strategies"""

    def __init__(self, depth_limit: int = 10):
        """Set the maximum depth this strategy will explore.

        Args:
            depth_limit: Maximum depth to explore in the search tree.
        """
        self.depth_limit = depth_limit

    @abstractmethod
    def search(self, root: TreeNode, llm: LLMInstance, **kwargs) -> Optional[TreeNode]:
        """Execute the search strategy.

        Args:
            root: Starting node of the search.
            llm: Language model used to generate or evaluate states.
            **kwargs: Strategy-specific parameters.

        Returns:
            The solution TreeNode if found, otherwise None
        """
        pass

    def _get_all_nodes(self, root: TreeNode) -> List[TreeNode]:
        """Get all nodes in the tree using BFS traversal.

        Args:
            root: Root node from which to start traversal.

        Returns:
            List of all nodes discovered in breadth-first order.
        """
        nodes = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            nodes.append(node)
            queue.extend(node.children)
        return nodes
