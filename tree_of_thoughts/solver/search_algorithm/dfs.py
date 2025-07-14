"""Depth-first search strategy"""
from typing import Callable, List, Optional

from tree_of_thoughts.tree_node import TreeNode
from tree_of_thoughts.llm_instance import LLMInstance
from .base import SearchStrategy


class DFSStrategy(SearchStrategy):
    """Depth-first search strategy following ToT-DFS algorithm"""

    def search(
        self,
        root: TreeNode,
        llm: LLMInstance,
        thought_generator,
        state_evaluator,
        solution_checker: Callable,
        k: int = 3,
        T: int = 3,
        v_th: float = 0.5,
        **kwargs
    ) -> Optional[TreeNode]:
        """Depth-first search returning the best solution node.

        Args:
            root: Root node of the search tree.
            llm: Language model instance for generation/evaluation.
            thought_generator: Callable that generates k thoughts from a node.
            state_evaluator: Callable that scores a list of candidate nodes.
            k: Number of thoughts to expand per state.
            T: Maximum search depth.
            v_th: Value threshold for pruning.
            solution_checker: Callable used to detect a valid solution.

        Returns:
            The best solution TreeNode found or None if none meet the criteria.
            
        """
        self.solutions: List[TreeNode] = []
        self.solution_checker = solution_checker
        self._dfs_recursive(root, 0, llm, thought_generator, state_evaluator, k, T, v_th)
        if self.solutions:
            return max(self.solutions, key=lambda s: s.value or 0)
        return None

    def _dfs_recursive(
        self,
        state: TreeNode,
        t: int,
        llm: LLMInstance,
        thought_generator,
        state_evaluator,
        k: int,
        T: int,
        v_th: float,
    ) -> None:
        """Recursive DFS step exploring children until depth T.

        Args:
            state: Current node being expanded.
            t: Current depth level in the search.
            llm: Language model instance.
            thought_generator: Callable generating new thoughts.
            state_evaluator: Callable scoring candidate nodes.
            k: Number of thoughts to generate per state.
            T: Maximum depth to explore.
            v_th: Value threshold for pruning branches.
        """
        if self.solution_checker(state):
            self.solutions.append(state)
            return

        if t >= T:
            final_thoughts = thought_generator(llm, state, 1)
            if final_thoughts:
                final_state = TreeNode(content=final_thoughts[0], depth=state.depth + 1, parent=state)
                state.add_child(final_state)
                if self.solution_checker(final_state):
                    self.solutions.append(final_state)
                else:
                    self.solutions.append(final_state)
            return

        thoughts = thought_generator(llm, state, k)
        candidates = []
        for thought in thoughts:
            child = TreeNode(content=thought, depth=state.depth + 1, parent=state)
            state.add_child(child)
            candidates.append(child)

        if candidates:
            evaluations = state_evaluator(llm, candidates)
            candidates.sort(key=lambda s: evaluations.get(s, 0), reverse=True)

            for candidate in candidates:
                evaluation_score = evaluations.get(candidate, 0)
                if evaluation_score > v_th:
                    self._dfs_recursive(candidate, t + 1, llm, thought_generator, state_evaluator, k, T, v_th)
                # If below threshold, prune this subtree
                else:
                    pass

