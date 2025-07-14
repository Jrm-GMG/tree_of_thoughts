"""A* search strategy"""
from typing import Callable, Dict, Tuple, Optional
import re

from tree_of_thoughts.tree_node import TreeNode
from tree_of_thoughts.llm_instance import LLMInstance
from .base import SearchStrategy
from ..reasoning.prompt import (
    VALIDITY_EVALUATION_PROMPT,
    COHERENCE_EVALUATION_PROMPT,
)


class AStarStrategy(SearchStrategy):
    """A* search guided by validity and coherence evaluation"""

    def __init__(self, depth_limit: int = 10, llm: Optional[LLMInstance] = None):
        """Create the strategy with a depth limit and LLM for scoring.

        Args:
            depth_limit: Maximum search depth to explore.
            llm: Language model used to score reasoning chains.
        """
        super().__init__(depth_limit)
        if not llm:
            raise ValueError("LLM instance is required for A* search strategy")
        self.llm = llm
        self.g_score_cache: Dict[str, float] = {}
        self.w_validity = 0.5
        self.w_coherence = 0.5
        # coefficient for how much to weigh the h score
        self.h_weight = 0.7

    def search(
        self,
        root: TreeNode,
        llm: LLMInstance,
        thought_generator,
        state_evaluator,
        solution_checker: Callable,
        **kwargs
    ) -> Optional[TreeNode]:
        """Run A* search returning a solution node or None.

        Args:
            root: Root node containing the initial problem.
            llm: Language model instance for generation/evaluation.
            thought_generator: Callable producing candidate thoughts.
            state_evaluator: Callable scoring a list of states.
            solution_checker: Callable used to detect a solution.

        Returns:
            The node containing a valid solution, or None if not found.
        """
        open_set = [root]
        closed_set = set()
        while open_set:
            current = min(open_set, key=self._calculate_f_score)
            open_set.remove(current)

            if solution_checker(current):
                return current
            node_id = id(current)
            if node_id in closed_set:
                continue
            closed_set.add(node_id)

            if current.depth >= self.depth_limit:
                continue

            thoughts = thought_generator(llm, current, 3)
            if thoughts:
                children = []
                for thought in thoughts:
                    child = TreeNode(content=thought, depth=current.depth + 1, parent=current)
                    current.add_child(child)
                    children.append(child)
                if children:
                    evaluations = state_evaluator(llm, children)
                    for child in children:
                        if id(child) not in closed_set:
                            open_set.append(child)
        return None

    def _calculate_f_score(self, node: TreeNode) -> float:
        """Combine g and h scores into an overall rank.

        Args:
            node: Node whose rank should be calculated.

        Returns:
            The computed f score for A* ordering.
        """
        g_score = self._get_g_score(node)
        h_score = self._get_h_score(node)
        g_weight = 1.0 - self.h_weight
        return g_weight * g_score + self.h_weight * h_score

    def _get_g_score(self, node: TreeNode) -> float:
        """Cost from start to node based on validity/coherence.

        Args:
            node: Node whose g score is requested.

        Returns:
            The path cost incorporating validity and coherence metrics.
        """
        cache_key = f"{node.content}_{node.depth}"
        if cache_key not in self.g_score_cache:
            validity, coherence = self._evaluate_chain_of_thought(node)
            g_score_raw = (
                self.w_validity * validity +
                self.w_coherence * coherence
            )
            g_score = 100 - g_score_raw
            self.g_score_cache[cache_key] = g_score
        return self.g_score_cache[cache_key] / 10

    def _get_h_score(self, node: TreeNode) -> float:
        """Heuristic estimate == the node's evaluation value.

        Args:
            node: Node whose heuristic cost should be estimated.

        Returns:
            The heuristic h score for the given node.
        """
        if node.value is not None:
            return node.value / 10
        return 5.0

    def _evaluate_chain_of_thought(self, node: TreeNode) -> Tuple[float, float]:
        """Return validity and coherence scores for a reasoning chain.

        Args:
            node: Node representing the end of the reasoning chain.

        Returns:
            Tuple of (validity_score, coherence_score).
        """
        path = node.get_path()
        if len(path) <= 1:
            return 50.0, 50.0
        reasoning_chain = "\n".join([f"Step {i}: {n.content}" for i, n in enumerate(path)])
        validity_score = self._evaluate_validity(reasoning_chain)
        coherence_score = self._evaluate_coherence(reasoning_chain)
        return validity_score, coherence_score

    def _evaluate_validity(self, reasoning_chain: str) -> float:
        """Score logical validity of a reasoning chain.

        Args:
            reasoning_chain: Combined reasoning steps to evaluate.

        Returns:
            Validity score on a 0-100 scale.
        """
        prompt = VALIDITY_EVALUATION_PROMPT.format(reasoning_chain=reasoning_chain)
        response = self.llm.generate(prompt, max_new_tokens=10, temperature=0.1, do_sample=False, num_return_sequences=1)[0]
        raw_score = self._extract_score(response, scale_max=4)
        return ((raw_score - 1) / 3) * 100

    def _evaluate_coherence(self, reasoning_chain: str) -> float:
        """Score coherence of a reasoning chain.

        Args:
            reasoning_chain: Combined reasoning steps to evaluate.

        Returns:
            Coherence score on a 0-100 scale.
        """
        prompt = COHERENCE_EVALUATION_PROMPT.format(reasoning_chain=reasoning_chain)
        response = self.llm.generate(prompt, max_new_tokens=10, temperature=0.1, do_sample=False, num_return_sequences=1)[0]
        raw_score = self._extract_score(response, scale_max=4)
        return ((raw_score - 1) / 3) * 100

    def _extract_score(self, response: str, scale_max: int = 4) -> float:
        """Parse score from pattern like grade{value/max}.

        Args:
            response: LLM response containing the score.
            scale_max: Maximum possible value in the scale used.

        Returns:
            Extracted numeric score as float. If parsing fails, returns the
            midpoint of the scale.
        """
        match = re.search(r"grade\{\s*(\d+(?:\.\d+)?)\s*/\s*(\d+)\s*\}", response, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return (scale_max + 1) / 2

