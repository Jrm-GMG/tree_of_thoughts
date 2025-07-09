"""Utility classes for presenting strategy comparison tables."""

from abc import ABC, abstractmethod
import pandas as pd
from .comparison import StrategyComparator

class BaseDoubleEntryTable(ABC):
    """Abstract base class for double entry tables."""

    @abstractmethod
    def build(self, comparator: StrategyComparator) -> pd.DataFrame:
        """Return DataFrame representing the table.

        Args:
            comparator: Strategy comparison object containing results.

        Returns:
            A pandas.DataFrame with the computed table.
        """
        pass

class SuccessOnlyTable(BaseDoubleEntryTable):
    """Table counting cases where row strategy succeeds and column fails."""

    def build(self, comparator: StrategyComparator) -> pd.DataFrame:
        """Return a pivot table of wins for each strategy pair.

        Args:
            comparator: Comparison data produced by :class:`StrategyComparator`.

        Returns:
            Pivot table showing how often each strategy wins over another.
        """
        df = comparator.export_comparison_matrix()
        table = (
            df.pivot(index='Strategy_A', columns='Strategy_B',
                     values='A_Only_Success').fillna(0)
        )
        return table.astype(int)

def build_comparator_from_results(results_by_strategy: dict) -> StrategyComparator:
    """Create :class:`StrategyComparator` from runner results.

    Args:
        results_by_strategy: Mapping from strategy name to run results.

    Returns:
        A populated :class:`StrategyComparator` instance.
    """
    comp = StrategyComparator()
    for strategy, data in results_by_strategy.items():
        for detail in data.get('details', []):
            comp.add_result(
                strategy_name=strategy,
                problem_id=detail.get('problem_id'),
                success=detail.get('success', False),
                solution=detail.get('solution'),
                correct_answer=detail.get('answer')
            )
    return comp
