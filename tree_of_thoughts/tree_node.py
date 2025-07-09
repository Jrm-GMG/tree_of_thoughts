"""
TreeNode data structure for Tree of Thoughts framework
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TreeNode:
    """Represents a node in the tree of thoughts"""
    
    # The actual thought content at this node
    content: str
    
    # Reference to parent node (None for root)
    parent: Optional['TreeNode'] = None
    
    # List of child nodes (subsequent thoughts)
    children: List['TreeNode'] = field(default_factory=list)
    
    # Evaluation score for this node (how promising it is)
    value: Optional[float] = None
    
    # How deep in the tree this node is (root = 0)
    depth: int = 0
    
    def add_child(self, child: 'TreeNode'):
        """Add a child node and set up relationships"""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
    
    def get_path(self) -> List['TreeNode']:
        """Get path from root to this node"""
        path = []
        node = self
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def __hash__(self):
        """Make nodes hashable (needed for sets/dicts)"""
        return hash(self.content)