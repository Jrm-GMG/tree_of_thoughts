"""
Main Tree of Thoughts solver - Updated to fix solution display issue
"""
from typing import Any, Optional, Tuple, Dict, List
from tree_of_thoughts.tree_node import TreeNode
from tree_of_thoughts.llm_instance import LLMInstance
from tree_of_thoughts.base_task import Task
from .generation import Generation
from .evaluation import Evaluation
from .enums import EvaluationMode
from ..search_algorithm.base import SearchStrategy
from ..prompts import get_root_node_content


class TreeOfThoughts:
    """Main Tree of Thoughts coordinator"""
    
    def __init__(
        self,
        task: Task,
        llm: LLMInstance,
        generation: Generation,
        evaluation: Evaluation,
        search_strategy: SearchStrategy,
        max_iterations: int = 50,
        # ToT parameters
        k: int = 3,  # Number of thoughts to generate per state
        T: int = 3,  # Maximum number of reasoning steps
        b: int = 5,  # Breadth limit for BFS
        v_th: float = 0.5  # Value threshold for DFS pruning
    ):
        """
        Initialize Tree of Thoughts solver
        
        Args:
            task: The task to solve
            llm: Language model instance
            generation: Module for generating thoughts
            evaluation: Module for evaluating thoughts
            search_strategy: How to explore the tree
            max_iterations: Maximum iterations to prevent infinite loops
            k: Number of thoughts to generate per state
            T: Maximum number of reasoning steps  
            b: Breadth limit (for BFS)
            v_th: Value threshold for pruning (for DFS)
        """
        self.task = task
        self.llm = llm
        self.generation = generation
        self.evaluation = evaluation
        self.search_strategy = search_strategy
        self.max_iterations = max_iterations
        
        # ToT algorithm parameters
        self.k = k
        self.T = T
        self.b = b
        self.v_th = v_th

        self.solution_node: Optional[TreeNode] = None
        
    def solve(self, problem_id: Any) -> Tuple[Optional[str], TreeNode]:
        """
        Solve a problem using Tree of Thoughts
        
        Args:
            problem_id: ID of the problem to solve
            
        Returns:
            Tuple of (solution, root_node) where solution is the answer or None
        """
        # Load the problem
        problem_data = self.task.load_problem(problem_id)
        task_prompt = self.task.get_task_prompt()
        
        # Initialize tree with root node containing the problem input
        root_content = get_root_node_content(problem_data)
        root = TreeNode(content=root_content, depth=0)
        
        # Create thought generator and state evaluator functions
        def thought_generator(llm: LLMInstance, state: TreeNode, k: int) -> List[str]:
            """Generate k candidate thoughts from current state"""
            return self.generation.generate(state, task_prompt, num_candidates=k)
        
        def state_evaluator(llm: LLMInstance, states: List[TreeNode]) -> Dict[TreeNode, float]:
            """Evaluate a list of states and return scores"""
            if self.evaluation.mode == EvaluationMode.VOTE:
                evaluations = self.evaluation.vote(states, task_prompt)
            else:
                evaluations = {}
                for state in states:
                    score = self.evaluation.evaluate(state, task_prompt)
                    evaluations[state] = score if score is not None else 0.0
                    state.value = score  # Also store in node
            return evaluations
        
        # Create solution checker function
        def solution_checker(node: TreeNode) -> bool:
            """Check if a node contains a valid solution"""
            return self.task.check_stopping_criteria(node)
        
        # Execute the search strategy with solution checker
        try:
            solution_node = self.search_strategy.search(
                root=root,
                llm=self.llm,
                thought_generator=thought_generator,
                state_evaluator=state_evaluator,
                solution_checker=solution_checker,  
                k=self.k,
                T=self.T,
                b=self.b,
                v_th=self.v_th
            )

            # Parse and return solution
            if solution_node:
                self.solution_node = solution_node
                solution = self.task.parse_solution(solution_node)
                return solution, root
                
        except Exception as e:
            print(f"Error during ToT search: {e}")
            
        return None, root
    
    
    def get_tree_stats(self, root: TreeNode) -> Dict[str, Any]:
        """Collect summary statistics from the search tree.

        Args:
            root: Root node of the search tree.

        Returns:
            Dictionary with aggregated statistics such as node counts,
            depth information and value ranges.
        """
        def _traverse_stats(node: TreeNode, stats: Dict):
            """Recursively accumulate statistics for node and children."""
            stats['total_nodes'] += 1
            stats['max_depth'] = max(stats['max_depth'], node.depth)
            
            if node.value is not None:
                stats['evaluated_nodes'] += 1
                stats['total_value'] += node.value
                stats['max_value'] = max(stats['max_value'], node.value)
                stats['min_value'] = min(stats['min_value'], node.value)
            
            if not node.children:
                stats['leaf_nodes'] += 1
            
            for child in node.children:
                _traverse_stats(child, stats)
        
        stats = {
            'total_nodes': 0,
            'evaluated_nodes': 0,
            'leaf_nodes': 0,
            'max_depth': 0,
            'total_value': 0.0,
            'max_value': float('-inf'),
            'min_value': float('inf')
        }
        
        _traverse_stats(root, stats)
        
        if stats['evaluated_nodes'] > 0:
            stats['avg_value'] = stats['total_value'] / stats['evaluated_nodes']
        else:
            stats['avg_value'] = 0.0
            
        if stats['max_value'] == float('-inf'):
            stats['max_value'] = None
        if stats['min_value'] == float('inf'):
            stats['min_value'] = None
            
        return stats
    
    def print_solution_path(self, solution_node: Optional[TreeNode] = None) -> None:
        """Print the path from the root to solution_node.

        Args:
            solution_node: The final node whose path should be displayed. If None,
                the most recent solution found.
        """
        node = solution_node or self.solution_node
        if node is None:
            print("No solution node available to display.")
            return

        path = node.get_path()

        print("\n=== Solution Path ===")
        for i, node in enumerate(path):
            value_str = f"{node.value:.2f}" if node.value is not None else "N/A"
            print(f"Step {i}: {node.content} (value: {value_str})")
    
