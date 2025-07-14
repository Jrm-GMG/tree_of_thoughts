"""
Game of 24 task implementation for Tree of Thoughts - Fixed Solution Parsing
"""

import re
from typing import Any, Dict, Optional
import pandas as pd

from tree_of_thoughts.base_task import Task
from tree_of_thoughts.tree_node import TreeNode
from .prompt import (
    get_game24_task_prompt,
    GAME24_FORMAT_PROMPT,
    GAME24_SOLUTION_MARKER,
)


class Game24Task(Task):
    """Implementation of the Game of 24 task with improved solution parsing"""
    
    def __init__(self, csv_path: str = "data/24/24.csv", debug: bool = False):
        """
        Initialize Game24 task
        
        Args:
            csv_path: Path to CSV file containing problems (default: data/24/24.csv)
            debug: Enable debug output
        """
        self.csv_path = csv_path
        self.problems_df = None
        self.current_problem = None
        self.target = 24
        self.debug = debug
        self._load_dataset()
    
    def _load_dataset(self):
        """Load and preprocess the CSV dataset"""
        import os
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
        self.problems_df = pd.read_csv(self.csv_path)
        
        # Convert percentage strings to floats only if they're still strings
        if 'Solved rate' in self.problems_df.columns:
            if self.problems_df['Solved rate'].dtype == 'object':
                self.problems_df['Solved rate'] = self.problems_df['Solved rate'].str.rstrip('%').astype(float) / 100
            elif self.problems_df['Solved rate'].dtype in ['float64', 'float32', 'int64', 'int32']:
                if (self.problems_df['Solved rate'] > 1).any():
                    self.problems_df['Solved rate'] = self.problems_df['Solved rate'] / 100
    
    def filter_by_difficulty(self, hard_only: bool = False, solve_rate_threshold: float = 0.30):
        """Filter problems by difficulty"""
        if hard_only and self.problems_df is not None:
            self.problems_df = self.problems_df[self.problems_df['Solved rate'] < solve_rate_threshold].reset_index(drop=True)
        
    def load_problem(self, problem_id: int) -> Dict[str, Any]:
        """Load a specific problem by ID"""
        if self.problems_df is None:
            raise ValueError("No dataset loaded")
            
        if problem_id >= len(self.problems_df):
            raise IndexError(f"Problem ID {problem_id} out of range (max: {len(self.problems_df)-1})")
            
        problem_row = self.problems_df.iloc[problem_id]
        
        # Parse numbers from the 'Puzzles' column
        numbers_str = problem_row['Puzzles']
        numbers = [int(x) for x in numbers_str.split()]
        
        # Convert string values to appropriate types
        self.current_problem = {
            'id': problem_id,
            'numbers': numbers,
            'rank': int(problem_row['Rank']),
            'difficulty_time': float(problem_row['AMT (s)']),
            'solve_rate': float(problem_row['Solved rate']),  
            'solution': None
        }
        return self.current_problem
    
    def get_task_prompt(self) -> str:
        """Return the prompt explaining the task"""
        if not self.current_problem:
            raise ValueError("No problem loaded")
            
        return get_game24_task_prompt(
            self.current_problem['numbers']
        )

    def get_format_prompt(self) -> str:
        """Return prompt for formatting the answer"""
        return GAME24_FORMAT_PROMPT

    def get_prompt_key(self) -> str:
        """Return generation/evaluation prompt key"""
        return 'game24'

    def _extract_solution_expression(self, text: str) -> Optional[str]:
        """Extract the expression appearing after the solution marker."""
        pattern = rf"{re.escape(GAME24_SOLUTION_MARKER)}\s*([^>]+)>>>"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return None

    def check_stopping_criteria(self, node: TreeNode) -> bool:
        """Return True if the node contains a valid final expression."""
        content = node.content

        if self.debug:
            print(f"\nDEBUG: Checking node (depth {node.depth}):")
            print(f"  Content preview: {content[:100]}...")

        expr = self._extract_solution_expression(content)
        if expr and self._validate_expression(expr):
            node._valid_expression = expr
            if self.debug:
                print(f"  OK Valid solution found: {expr}")
            return True

        if self.debug:
            print("  X No valid solution found")

        return False

    def _validate_expression(self, expr: str) -> bool:
        """Return True if expression uses the numbers exactly once and equals 24."""
        if not self.current_problem or not expr:
            return False

        try:
            expr = re.sub(r'[^\d\+\-\*/\(\)\s]', '', expr)
            nums_in_expr = [int(n) for n in re.findall(r'\d+', expr)]
            required_nums = sorted(self.current_problem['numbers'])
            if sorted(nums_in_expr) != required_nums:
                if self.debug:
                    print(f"    Numbers mismatch: got {sorted(nums_in_expr)}, expected {required_nums}")
                return False
            result = eval(expr)
            is_valid = abs(result - self.target) < 1e-6
            if self.debug:
                print(f"    Evaluated: {expr} = {result}, valid: {is_valid}")
            return is_valid
        except Exception as e:
            if self.debug:
                print(f"    Validation error for '{expr}': {e}")
            return False

    def parse_solution(self, node: TreeNode) -> str:
        """Return the verified solution expression from a node."""
        if hasattr(node, '_valid_expression'):
            return node._valid_expression

        if self.debug:
            print(f"\nDEBUG: Parsing solution from node (depth {node.depth})")

        expr = self._extract_solution_expression(node.content)
        if expr and self._validate_expression(expr):
            if self.debug:
                print(f"  Parsed solution: {expr}")
            return expr

        if self.debug:
            print("  Failed to parse solution")

        return ""
    
    def get_total_problems(self) -> int:
        """Return the total number of problems available"""
        return len(self.problems_df) if self.problems_df is not None else 0
    
    def get_problem_info(self, problem_id: int) -> Dict[str, Any]:
        """Get additional information about a problem"""
        if self.problems_df is None:
            raise ValueError("No dataset loaded")
            
        if problem_id >= len(self.problems_df):
            raise IndexError(f"Problem ID {problem_id} out of range")
            
        row = self.problems_df.iloc[problem_id]
        return {
            'rank': int(row['Rank']),
            'numbers': row['Puzzles'],
            'avg_time': float(row['AMT (s)']),
            'solve_rate': float(row['Solved rate']),  
            'mean_time': float(row['1-sigma Mean (s)']),
            'std_time': float(row['1-sigma STD (s)'])        }