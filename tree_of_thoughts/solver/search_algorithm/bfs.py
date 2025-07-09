"""Breadth-first search strategy"""
from typing import Callable, List, Optional

from tree_of_thoughts.tree_node import TreeNode
from tree_of_thoughts.llm_instance import LLMInstance
from .base import SearchStrategy


class BFSStrategy(SearchStrategy):
    """Breadth-first search strategy following ToT-BFS algorithm"""

    def search(
        self,
        root: TreeNode,
        llm: LLMInstance,
        thought_generator,
        state_evaluator,
        solution_checker: Callable,
        k: int = 3,
        T: int = 3,
        b: int = 5,
        **kwargs
    ) -> Optional[TreeNode]:
        """Execute ToT-BFS algorithm"""
        S = [root]

        for _ in range(1, T + 1):
            S_prime = []
            for s in S:
                thoughts = thought_generator(llm, s, k)
                for thought in thoughts:
                    child = TreeNode(content=thought, depth=s.depth + 1, parent=s)
                    s.add_child(child)
                    S_prime.append(child)

            if not S_prime:
                break

            V_t = state_evaluator(llm, S_prime)

            for state in S_prime:
                if solution_checker(state):
                    return state

            evaluated_states = [(state, V_t.get(state, 0)) for state in S_prime]
            evaluated_states.sort(key=lambda x: x[1], reverse=True)
            S = [state for state, _ in evaluated_states[:b]]

        if S:
            for state in S:
                if solution_checker(state):
                    return state

            final_states = []
            for state in S:
                final_thoughts = thought_generator(llm, state, 1)
                if final_thoughts:
                    final_child = TreeNode(content=final_thoughts[0], depth=state.depth + 1, parent=state)
                    state.add_child(final_child)
                    final_states.append(final_child)

            if final_states:
                for state in final_states:
                    if solution_checker(state):
                        return state
                final_V = state_evaluator(llm, final_states)
                return max(final_states, key=lambda s: final_V.get(s, 0))

        return None
