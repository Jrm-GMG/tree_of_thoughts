"""
Strategy comparison utilities
"""

from typing import Dict, List, Any
import pandas as pd


class StrategyComparator:
    """Handles detailed comparison between strategies"""

    def __init__(self):
        """Initialize empty storage for results and comparisons."""
        self.results = {}
        self.comparisons = {}
    
    def add_result(self, strategy_name: str, problem_id: int, success: bool,
                   solution: str = None, correct_answer: str = None, **kwargs):
        """Add a result for a strategy.

        Args:
            strategy_name: Name of the strategy being evaluated.
            problem_id: Identifier of the problem instance.
            success: True if the strategy solved the problem.
            solution: Solution string produced by the strategy.
            correct_answer: Reference answer for the problem.
            **kwargs: Additional metadata to store with the result.
        """
        if strategy_name not in self.results:
            self.results[strategy_name] = {}
        
        self.results[strategy_name][problem_id] = {
            'success': success,
            'solution': solution,
            'correct_answer': correct_answer,
            **kwargs
        }
    
    def compare_strategies(self, strategy1: str, strategy2: str) -> Dict[str, int]:
        """Compare two strategies and return detailed statistics.

        Args:
            strategy1: Name of the first strategy.
            strategy2: Name of the second strategy.

        Returns:
            Dictionary summarizing agreement and success metrics.
        """
        comparison = {
            'both_success': 0,
            'both_failed': 0,
            'strategy1_only_success': 0,
            'strategy2_only_success': 0,
            'total_problems': 0,
            'both_correct_same': 0,
            'both_correct_different': 0,
            'strategy1_correct_only': 0,
            'strategy2_correct_only': 0
        }
        
        # Get common problem IDs
        problems1 = set(self.results.get(strategy1, {}).keys())
        problems2 = set(self.results.get(strategy2, {}).keys())
        common_problems = problems1.intersection(problems2)
        
        for problem_id in common_problems:
            result1 = self.results[strategy1][problem_id]
            result2 = self.results[strategy2][problem_id]
            
            success1 = result1['success']
            success2 = result2['success']
            solution1 = result1.get('solution', '')
            solution2 = result2.get('solution', '')
            
            # Basic success/failure comparison
            if success1 and success2:
                comparison['both_success'] += 1
                # Check if solutions are the same
                if solution1 and solution2:
                    if solution1.strip() == solution2.strip():
                        comparison['both_correct_same'] += 1
                    else:
                        comparison['both_correct_different'] += 1
            elif not success1 and not success2:
                comparison['both_failed'] += 1
            elif success1 and not success2:
                comparison['strategy1_only_success'] += 1
                comparison['strategy1_correct_only'] += 1
            elif not success1 and success2:
                comparison['strategy2_only_success'] += 1
                comparison['strategy2_correct_only'] += 1
            
            comparison['total_problems'] += 1
        
        # Store for later reference
        comparison_key = f"{strategy1}_vs_{strategy2}"
        self.comparisons[comparison_key] = comparison
        
        return comparison
    
    def get_all_strategies(self) -> List[str]:
        """Get list of all strategies that have results."""
        return list(self.results.keys())
    
    def export_comparison_matrix(self) -> pd.DataFrame:
        """Export comparison data as a matrix for analysis.

        Returns:
            A pandas.DataFrame where each row compares two strategies.
        """
        strategies = self.get_all_strategies()
        matrix_data = []
        
        for strategy1 in strategies:
            for strategy2 in strategies:
                if strategy1 != strategy2:
                    comparison = self.compare_strategies(strategy1, strategy2)
                    matrix_data.append({
                        'Strategy_A': strategy1,
                        'Strategy_B': strategy2,
                        'Both_Success': comparison['both_success'],
                        'Both_Failed': comparison['both_failed'],
                        'A_Only_Success': comparison['strategy1_only_success'],
                        'B_Only_Success': comparison['strategy2_only_success'],
                        'Both_Same_Solution': comparison['both_correct_same'],
                        'Both_Different_Solution': comparison['both_correct_different'],
                        'Total_Problems': comparison['total_problems'],
                        'Agreement_Rate': (comparison['both_success'] + comparison['both_failed']) / comparison['total_problems'] * 100 if comparison['total_problems'] > 0 else 0,
                        'Complementary_Rate': (comparison['strategy1_only_success'] + comparison['strategy2_only_success']) / comparison['total_problems'] * 100 if comparison['total_problems'] > 0 else 0
                    })
        
        return pd.DataFrame(matrix_data)
    
