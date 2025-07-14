"""
Generation module for Tree of Thoughts framework
"""
import re
from typing import List, Optional
from .enums import GenerationMode
from tree_of_thoughts.tree_node import TreeNode
from tree_of_thoughts.llm_instance import LLMInstance
from .prompt import (
    GENERATION_SYSTEM_PROMPTS,
    get_generation_user_prompt,
)
from transformers.utils import logging
logging.set_verbosity_error()


class Generation:
    """Handles LLM generation for creating new thoughts"""
    
    def __init__(
        self,
        llm: LLMInstance,
        system_prompt: str = "",
        mode: GenerationMode = GenerationMode.SAMPLE,
        num_generations: int = 3,
        temperature: float = 0.7,
        max_new_tokens: int = 200,
        prompt_key: str = 'default'
    ):
        """
        Initialize the Generation module
        
        Args:
            llm: LLM instance to use for generation
            system_prompt: System instructions for the LLM (overrides default if provided)
            mode: Generation mode (SAMPLE or PROPOSE)
            num_generations: Number of thoughts to generate
            temperature: Generation randomness
            max_new_tokens: Length limit for each thought
            prompt_key: Key to select default system prompt from GENERATION_SYSTEM_PROMPTS
        """
        self.llm = llm
        self.system_prompt = system_prompt or GENERATION_SYSTEM_PROMPTS.get(prompt_key, GENERATION_SYSTEM_PROMPTS['default'])
        self.mode = mode
        self.num_generations = num_generations
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
    
    def generate(self, node: TreeNode, task_prompt: str, num_candidates: Optional[int] = None) -> List[str]:
        """
        Generate new thoughts from current node
        
        Args:
            node: Current node in the tree
            task_prompt: The task description
            num_candidates: Number of candidates to generate (overrides default if provided)
            
        Returns:
            List of generated thoughts
        """
        num_to_generate = num_candidates if num_candidates is not None else self.num_generations
        
        path = node.get_path()
        context = "\n".join([f"Step {i}: {n.content}" for i, n in enumerate(path)])
        
        # Get prompt 
        user_prompt = get_generation_user_prompt(
            self.mode.value,
            context,
            num_to_generate
        )
        if task_prompt:
            user_prompt = f"{task_prompt}\n\n{user_prompt}"
        
        full_prompt = f"{self.system_prompt}\n\n{user_prompt}\n\nResponse:"
        
        if self.mode == GenerationMode.PROPOSE:
            # Single generation that contains multiple proposals
            outputs = self.llm.generate(
                full_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                num_return_sequences=1
            )
            
            content = outputs[0]
            proposals = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', content, re.DOTALL)
            return [p.strip() for p in proposals[:num_to_generate]]
        
        else:  # SAMPLE mode
            # Generate multiple independent completions
            outputs = self.llm.generate(
                full_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                num_return_sequences=num_to_generate
            )
            return [out.strip() for out in outputs]
    
    
