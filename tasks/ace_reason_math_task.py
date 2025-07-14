"""
AceReason-Math task implementation for Tree of Thoughts framework
"""

import re
import sympy as sp
from datasets import load_dataset
from typing import Dict, List, Optional, Any
import random

from tree_of_thoughts.base_task import Task
from tree_of_thoughts.tree_node import TreeNode
from .prompt import get_ace_math_task_prompt, ACE_MATH_FORMAT_PROMPT


class AceReasonMathTask(Task):
    """
    Task class for NVIDIA AceReason-Math dataset integration with Tree of Thoughts framework.
    Provides methods for loading problems, generating prompts, verifying solutions,
    and parsing model outputs.
    """
    
    def __init__(self, streaming: bool = False, debug: bool = False):
        """
        Initialize the AceReason-Math task.
        
        Args:
            streaming: Whether to load dataset in streaming mode for memory efficiency
            debug: Enable debug output
        """
        self.dataset_name = "nvidia/AceReason-Math"
        self.streaming = streaming
        self.debug = debug
        
        if streaming:
            self.dataset = load_dataset(self.dataset_name, streaming=True)["train"]
            self.problems = None  
            self.total_problems = len(load_dataset(self.dataset_name, split="train"))
        else:
            self.dataset = load_dataset(self.dataset_name)["train"]
            self.problems = {i: self.dataset[i] for i in range(len(self.dataset))}
            self.total_problems = len(self.dataset)
        self.current_problem = None
        
    def load_problem(self, problem_id: int) -> Dict[str, Any]:
        """
        Load a specific problem by its ID.
        
        Args:
            problem_id: Integer ID of the problem (0-based index)
            
        Returns:
            Dictionary containing problem data
        """
        if problem_id < 0 or problem_id >= self.total_problems:
            raise ValueError(f"Problem ID must be between 0 and {self.total_problems-1}")
            
        if self.streaming:
            for i, item in enumerate(self.dataset):
                if i == problem_id:
                    problem_data = item
                    break
        else:
            problem_data = self.problems[problem_id]
        
        self.current_problem = {
            'id': problem_id,
            'problem': problem_data['problem'],
            'answer': problem_data['answer']
        }
        
        return self.current_problem
    
    def get_task_prompt(self) -> str:
        """Return the prompt explaining the task"""
        if not self.current_problem:
            raise ValueError("No problem loaded")
            
        return get_ace_math_task_prompt(self.current_problem['problem'])

    def get_format_prompt(self) -> str:
        """Return prompt for formatting the answer"""
        return ACE_MATH_FORMAT_PROMPT

    def get_prompt_key(self) -> str:
        """Return generation/evaluation prompt key"""
        return 'math'

    def check_stopping_criteria(self, node: TreeNode) -> bool:
        """Check if the current node contains a valid solution"""
        content = node.content
        
        boxed_answer = self._extract_boxed_answer(content)

        if self.debug:
            print(f"\nDEBUG: Checking node (depth {node.depth}):")
            if boxed_answer is not None:
                print(f"  Extracted answer: {boxed_answer}")
                if self.current_problem:
                    print(f"  Correct answer: {self.current_problem['answer']}")
            else:
                print("  X No \\boxed{} answer found")

        if boxed_answer is None:
            return False
        
        # Verify the answer is mathematically correct
        if self.current_problem and self._verify_mathematical_equivalence(
            boxed_answer,
            self.current_problem['answer']
        ):
            # Store answer
            node._valid_answer = boxed_answer
            if self.debug:
                print(f"  OK Valid solution found: {boxed_answer}")
            return True

        if self.debug:
            print(f"  X Answer {boxed_answer} is incorrect")

        return False

    def parse_solution(self, node: TreeNode) -> Any:
        """Parse the final solution from a node"""
        # check if we stored a valid answer during stopping criteria check
        if hasattr(node, '_valid_answer'):
            return node._valid_answer
        
        if self.debug:
            print(f"\nDEBUG: Parsing solution from node (depth {node.depth})")

        # Otherwise, try to extract it again
        answer = self._extract_boxed_answer(node.content)

        if answer and self.current_problem and self._verify_mathematical_equivalence(
            answer,
            self.current_problem['answer']
        ):
            if self.debug:
                print(f"  Extracted answer: {answer}")
                print(f"  Correct answer: {self.current_problem['answer']}")
            return answer

        if self.debug:
            if answer is not None:
                print(f"  Extracted answer: {answer}")
                if self.current_problem:
                    print(f"  Correct answer: {self.current_problem['answer']}")
            print("  Failed to parse valid solution")
                
        return None
    
    def _extract_boxed_answer(self, text: str) -> Optional[str]:
        """
        Extract the final answer from text in \\boxed{} format.
        
        Args:
            text: String containing the response
            
        Returns:
            Extracted answer string or None if no valid format found
        """
        # Pattern to match \boxed{content} 
        boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(boxed_pattern, text)
        
        if matches:
            # Return the last \boxed{} answer 
            return matches[-1].strip()
        return None
    
    def _verify_mathematical_equivalence(self, predicted: str, correct: str) -> bool:
        """
        Verify mathematical equivalence between predicted and correct answers.
        
        Args:
            predicted: Predicted answer string
            correct: Correct answer string
            
        Returns:
            Boolean indicating mathematical equivalence
        """
        try:
            #symbolic mathematics comparison
            predicted_expr = sp.sympify(predicted)
            correct_expr = sp.sympify(correct)
            
            # Check if the difference simplifies to zero
            difference = sp.simplify(predicted_expr - correct_expr)
            return difference == 0
            
        except (sp.SympifyError, ValueError, TypeError, ZeroDivisionError):
            # Fallback to string comparison for non-symbolic answers or when
            # sympy fails to parse the expressions 
            return str(predicted).strip().lower() == str(correct).strip().lower()
    
    def get_total_problems(self) -> int:
        """Return the total number of problems available"""
        return self.total_problems
    
    def get_problem_info(self, problem_id: int) -> Dict[str, Any]:
        """Get additional information about a problem"""
        if problem_id < 0 or problem_id >= self.total_problems:
            raise ValueError(f"Problem ID must be between 0 and {self.total_problems-1}")
            
        if self.streaming:
            for i, item in enumerate(self.dataset):
                if i == problem_id:
                    problem_data = item
                    break
        else:
            problem_data = self.problems[problem_id]
        
        return {
            'id': problem_id,
            'problem_preview': problem_data['problem'][:100] + '...' if len(problem_data['problem']) > 100 else problem_data['problem'],
            'answer': problem_data['answer'],
            'problem_length': len(problem_data['problem']),
            'answer_length': len(problem_data['answer'])
        }
