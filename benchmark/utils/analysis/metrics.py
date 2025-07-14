"""Analysis utilities for demo results"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import json
from ..helpers import ensure_directory_exists
from .double_entry import build_comparator_from_results


class ResultAnalyzer:
    """Analyzes and compares results across strategies"""

    def __init__(self, results: Dict[str, Dict[str, Any]]):
        """Initialize with raw results indexed by strategy.

        Args:
            results: Mapping of strategy name to its result dictionary.
        """
        self.results = results
        self.strategies = list(results.keys())
    
    def create_comparison_matrix(self) -> pd.DataFrame:
        """Create detailed comparison matrix.

        Returns:
            A pandas.DataFrame summarizing performance metrics per strategy.
        """
        data = []
        
        for strategy in self.strategies:
            result = self.results[strategy]
            details = result.get('details', [])
            
            row = {
                'Strategy': strategy,
                'Success Rate': result['success_rate'],
                'Avg Time (s)': result['avg_time'],
                'Total Solved': result['total_solved'],
                'Total Problems': result['total_problems'],
            }

            # Add strategy-specific metrics
            if details and 'nodes_explored' in details[0]:
                row['Avg Nodes'] = np.mean([d.get('nodes_explored', 0) for d in details])
            else:
                row['Avg Nodes'] = None
            
            data.append(row)
        
        df = pd.DataFrame(data).set_index('Strategy')
        return df.fillna('None')
    
    def _build_comparator(self):
        """Helper to construct a :class:`StrategyComparator` from stored results."""
        return build_comparator_from_results(self.results)

    def pairwise_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Perform pairwise strategy comparison using :class:`StrategyComparator`.

        Returns:
            Mapping from strategy pair to aggregated comparison metrics.
        """
        comp = self._build_comparator()
        strategies = comp.get_all_strategies()
        comparisons = {}

        for i, strat1 in enumerate(strategies):
            for strat2 in strategies[i + 1 :]:
                stats = comp.compare_strategies(strat1, strat2)
                key = f"{strat1}_vs_{strat2}"

                both_solved = stats['both_success']
                strat1_only = stats['strategy1_only_success']
                strat2_only = stats['strategy2_only_success']
                neither = stats['both_failed']

                comparisons[key] = {
                    'both_solved': both_solved,
                    f'{strat1}_only': strat1_only,
                    f'{strat2}_only': strat2_only,
                    'neither_solved': neither,
                    'agreement_rate': (
                        both_solved + neither
                    )
                    / stats['total_problems']
                    if stats['total_problems']
                    else 0,
                }

        return comparisons
    
    def get_best_strategy(self, metric: str = 'success_rate') -> str:
        """Get best strategy by metric.

        Args:
            metric: Metric key to sort strategies by (default 'success_rate').

        Returns:
            Name of the strategy with the highest value for metric.
        """
        return max(self.results.items(), key=lambda x: x[1].get(metric, 0))[0]
    
    def export_analysis(self, filepath: str):
        """Export complete analysis to file.

        Args:
            filepath: Destination JSON file path.
        """
        analysis = {
            'summary': self.create_comparison_matrix().to_dict(),
            'pairwise': self.pairwise_comparison(),
            'best_by_success': self.get_best_strategy('success_rate'),
            'best_by_speed': min(self.results.items(), 
                               key=lambda x: x[1].get('avg_time', float('inf')))[0]
        }
        
        ensure_directory_exists(filepath)
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
