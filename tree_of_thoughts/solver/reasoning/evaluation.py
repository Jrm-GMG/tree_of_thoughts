"""
Evaluation module for Tree of Thoughts framework
"""

import re
from typing import Dict, List

from .enums import EvaluationMode
from tree_of_thoughts.tree_node import TreeNode
from tree_of_thoughts.llm_instance import LLMInstance
from .prompt import (
    EVALUATION_SYSTEM_PROMPTS,
    get_evaluation_user_prompt,
    get_vote_prompt,
)


class Evaluation:
    """Handles evaluation of nodes using LLM"""
    
    def __init__(
        self,
        llm: LLMInstance,
        system_prompt: str = "",
        mode: EvaluationMode = EvaluationMode.VALUE,
        temperature: float = 0.1,
        max_new_tokens: int = 50,
        prompt_key: str = 'default'
    ):
        """
        Initialize the Evaluation module
        
        Args:
            llm: LLM instance to use for evaluation
            system_prompt: System instructions for the LLM (overrides default if provided)
            mode: Evaluation mode (VALUE or VOTE)
            temperature: Generation temperature (low for consistency)
            max_new_tokens: Maximum tokens for evaluation response
            prompt_key: Key to select default system prompt from EVALUATION_SYSTEM_PROMPTS
        """
        self.llm = llm
        # Use provided system_prompt or fall back to default
        self.system_prompt = system_prompt or EVALUATION_SYSTEM_PROMPTS.get(prompt_key, EVALUATION_SYSTEM_PROMPTS['default'])
        self.mode = mode
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.cache: Dict[str, float] = {}

    def vote(self, nodes: List[TreeNode], task_prompt: str, target: str = "") -> Dict[TreeNode, float]:
        """Evaluate multiple nodes at once using the ToT voting scheme"""
        if not nodes:
            return {}

        choices = [node.content for node in nodes]
        user_prompt = get_vote_prompt(task_prompt, choices)

        full_prompt = f"{self.system_prompt}\n\n{user_prompt}\n\nResponse:"

        outputs = self.llm.generate(
            full_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=False,
            num_return_sequences=1,
        )

        content = outputs[0].strip()
        match = re.search(r"The best choice is\s*(\d+)", content, re.IGNORECASE)
        if match:
            try:
                best = int(match.group(1)) - 1
            except ValueError:
                best = 0
        else:
            best = 0

        results: Dict[TreeNode, float] = {}
        for idx, node in enumerate(nodes):
            value = 100.0 if idx == best else 0.0
            node.value = value
            results[node] = value
        return results
    
    def evaluate(self, node: TreeNode, task_prompt: str, target: str = "") -> float:
        """
        Evaluate a node's promise/value
        
        Args:
            node: Node to evaluate
            task_prompt: The task description
            target: Optional target value/goal
            
        Returns:
            Evaluation score (0-100 for VALUE mode, 0 or 100 for VOTE mode)
        """
        # Check cache 
        cache_key = f"{node.content}_{node.depth}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
    
        path = node.get_path()
        context = "\n".join([f"Step {i}: {n.content}" for i, n in enumerate(path)])
        
        # prompt 
        user_prompt = get_evaluation_user_prompt(
            self.mode.value,
            context,
            target
        )
        if task_prompt:
            user_prompt = f"{task_prompt}\n\n{user_prompt}"
        
        full_prompt = f"{self.system_prompt}\n\n{user_prompt}\n\nResponse:"
        
        # Generate evaluation
        outputs = self.llm.generate(
            full_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=False,
            num_return_sequences=1
        )
        
        content = outputs[0].strip()

        if self.mode == EvaluationMode.VALUE:
            match = re.search(r'grade\s*\{?(\d+(?:\.\d+)?)\}?', content, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    value = min(100.0, max(0.0, value))
                except ValueError:
                    value = 50.0
            else:
                value = 50.0
        else:  # VOTE mode
            content_upper = content.upper()
            if any(word in content_upper for word in ["CONTINUE", "YES", "KEEP"]):
                value = 100.0
            elif any(word in content_upper for word in ["ABANDON", "NO", "DISCARD"]):
                value = 0.0
            else:
                value = 0.0
        
        # Cache and store result
        self.cache[cache_key] = value
        node.value = value
        
        return value
