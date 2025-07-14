"""Base runner classes for benchmarks"""

import time
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from tqdm import tqdm
from ..helpers import ensure_directory_exists

from tree_of_thoughts.solver.reasoning.tree_of_toughts import TreeOfThoughts
from tree_of_thoughts.solver.search_algorithm.bfs import BFSStrategy
from tree_of_thoughts.solver.search_algorithm.dfs import DFSStrategy
from tree_of_thoughts.solver.search_algorithm.astar import AStarStrategy
from tree_of_thoughts.solver.reasoning.enums import GenerationMode, EvaluationMode, SearchStrategy as SearchStrategyEnum


class BaseRunner(ABC):
    """Base class for demo runners"""
    
    def __init__(self, config: Dict[str, Any], llm, task):
        """Store run configuration, LLM and task."""
        self.config = config
        self.llm = llm
        self.task = task
        self.results = []
    
    @abstractmethod
    def run(self, dataset: Union[pd.DataFrame, Dict[str, Any], None]) -> Dict[str, Any]:
        """Run the demo and return results"""
        pass
    
    def create_search_strategy(self, strategy_name: str):
        """Create search strategy from configuration"""
        search_config = self.config.get('search', {})
        
        if strategy_name == 'bfs':
            config = search_config.get('bfs', {})
            return BFSStrategy(depth_limit=config.get('depth_limit', 4))
        
        elif strategy_name == 'dfs':
            config = search_config.get('dfs', {})
            return DFSStrategy(depth_limit=config.get('depth_limit', 4))
        
        elif strategy_name == 'astar':
            config = search_config.get('astar', {})
            return AStarStrategy(
                depth_limit=config.get('depth_limit', 4),
                llm=self.llm
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def save_results(self, results: Dict[str, Any]):
        """Save results based on configuration"""
        output_config = self.config.get('output', {})
        
        # Collect all details from all strategies
        all_details = []
        for strategy_name, strategy_results in results.get('strategies', {}).items():
            for detail in strategy_results.get('details', []):
                detail_copy = detail.copy()
                detail_copy['strategy'] = strategy_name
                all_details.append(detail_copy)
        
        if 'results' in output_config and all_details:
            df = pd.DataFrame(all_details)
            ensure_directory_exists(output_config['results'])
            df.to_csv(output_config['results'], index=False)
            print(f"\nResults saved to: {output_config['results']}")


class ComparisonRunner(BaseRunner):
    """Runner for comparing multiple strategies"""
    
    def run(self, dataset: Union[pd.DataFrame, Dict[str, Any], None]) -> Dict[str, Any]:
        """Run comparison across strategies"""
        strategies = self.config.get('strategies', ['baseline_cot', 'bfs'])
        
        # Determine number of problems based on dataset type
        if isinstance(dataset, pd.DataFrame):
            num_problems = min(self.config.get('num_puzzles', self.config.get('num_problems', 5)), len(dataset))
        elif isinstance(dataset, dict):
            # For HuggingFace datasets with metadata
            num_problems = len(dataset.get('problem_ids', []))
            if num_problems == 0:
                num_problems = dataset.get('num_problems', self.config.get('num_problems', 5))
        else:
            num_problems = self.config.get('num_problems', 5)
        
        results_by_strategy = {}
        
        for strategy_name in strategies:
            
            if strategy_name.startswith('baseline'):
                baseline_type = 'direct' if strategy_name == 'baseline_direct' else 'cot'
                results = self._run_baseline(dataset, num_problems, baseline_type)
            else:
                results = self._run_tot(dataset, num_problems, strategy_name)
            
            results_by_strategy[strategy_name] = results
        
        # Analyze comparison
        comparison = self._analyze_comparison(results_by_strategy)
        
        return {
            'strategies': results_by_strategy,
            'comparison': comparison,
            'details': self._collect_all_details(results_by_strategy)
        }
    
    def _run_baseline(
        self,
        dataset: Union[pd.DataFrame, Dict[str, Any], None],
        num_problems: int,
        baseline_type: str = 'cot'
    ) -> Dict[str, Any]:
        """Run baseline solver.

        Supports an *oracle* configuration where ``oracle_k`` independent runs
        are performed for each problem and the best result is kept.
        """
        baseline_temp = self.config.get('baseline', {}).get('temperature', 1.0)
        if self.config.get('task') == 'game24':
            if baseline_type == 'direct':
                from baseline.game24.game24_direct_solver import Game24DirectSolver
                solver = Game24DirectSolver(self.llm, self.task, baseline_temp)
            else:
                from baseline.game24.game24_baseline_solver import Game24BaselineSolver
                solver = Game24BaselineSolver(self.llm, self.task, baseline_temp)
        elif self.config.get('task') == 'math':
            if baseline_type == 'direct':
                from baseline.math.math_direct_solver import MathDirectSolver
                solver = MathDirectSolver(self.llm, self.task, baseline_temp)
            else:
                from baseline.math.math_baseline_solver import MathBaselineSolver
                solver = MathBaselineSolver(self.llm, self.task, baseline_temp)
        elif self.config.get('task') == 'gsm8k':
            if baseline_type == 'direct':
                from baseline.gsm8k.gsm8k_direct_solver import GSM8KDirectSolver
                solver = GSM8KDirectSolver(self.llm, self.task, baseline_temp)
            else:
                from baseline.gsm8k.gsm8k_baseline_solver import GSM8KBaselineSolver
                solver = GSM8KBaselineSolver(self.llm, self.task, baseline_temp)
        else:
            raise ValueError(f"No baseline solver for task: {self.config.get('task')}")
        
        results = []
        strategy_name = 'baseline_direct' if baseline_type == 'direct' else 'baseline_cot'
        oracle_k = self.config.get('baseline', {}).get('oracle_k', 1)

        progress_desc = f"Testing {strategy_name.upper()} Strategy"
        for i in tqdm(range(num_problems), desc=progress_desc, unit='problem'):
            problem_data = self._get_problem(dataset, i)
            start_time = time.time()
            best_solution = None
            best_time = float('inf')
            total_attempts = 0

            for _ in range(max(oracle_k, 1)):
                sol, t_taken, atts = solver.solve(problem_data['id'])
                valid = sol is not None

                total_attempts += atts

                if valid and t_taken < best_time:
                    best_solution = sol
                    best_time = t_taken

            success = best_solution is not None
            
            results.append({
                'problem_id': problem_data.get('id', i),
                'success': success,
                'time': best_time if success else time.time() - start_time,
                'solution': best_solution,
                'answer': problem_data.get('answer'),
                'attempts': total_attempts
            })
            
            if self.config.get('verbose'):
                status = 'OK' if success else 'X'
                predicted = best_solution if best_solution is not None else "None"
                answer = problem_data.get('answer')
                print(
                    f"Problem {i+1}: {status} "
                    f"(attempts: {total_attempts}) - Predicted: {predicted} / "
                    f"Answer: {answer}"
                )
        
        return self._summarize_results(results)
    
    def _run_tot(self, dataset: Union[pd.DataFrame, Dict[str, Any], None], num_problems: int, 
                 strategy_name: str) -> Dict[str, Any]:
        """Run Tree of Thoughts solver"""
        from ..config.setup import create_tot_components
        
        generation, evaluation = create_tot_components(self.llm, self.task, self.config)
        search_strategy = self.create_search_strategy(strategy_name)
        
        # ToT parameters from config
        tot_config = self.config.get('tot', {})
        
        solver = TreeOfThoughts(
            task=self.task,
            llm=self.llm,
            generation=generation,
            evaluation=evaluation,
            search_strategy=search_strategy,
            max_iterations=tot_config.get('max_iterations', 50),
            k=tot_config.get('k', 3),
            T=tot_config.get('T', 3),
            b=tot_config.get('b', 5),
            v_th=tot_config.get('v_th', 0.5)
        )
        
        results = []

        progress_desc = f"Testing {strategy_name.upper()} Strategy"
        for i in tqdm(range(num_problems), desc=progress_desc, unit='problem'):
            problem_data = self._get_problem(dataset, i)
            problem_id = problem_data.get('id', i)
            
            start_time = time.time()
            
            # Load problem into task
            self.task.load_problem(problem_id)
            
            # Solve using ToT
            solution, tree_root = solver.solve(problem_id)
            success = solution is not None
            
            # Get tree statistics
            tree_stats = solver.get_tree_stats(tree_root)
            
            results.append({
                'problem_id': problem_id,
                'success': success,
                'time': time.time() - start_time,
                'solution': solution,
                'answer': problem_data.get('answer'),
                'nodes_explored': tree_stats['total_nodes'],
                'max_depth': tree_stats['max_depth']
            })
            
            if self.config.get('verbose'):
                status = 'OK' if success else 'X'
                predicted = solution if solution is not None else "None"
                answer = problem_data.get('answer')
                print(
                    f"Problem {i+1}: {status} (nodes: {tree_stats['total_nodes']}) "
                    f"- Predicted: {predicted} / Answer: {answer}"
                )
            
            if self.config.get('show_trees') and success:
                print(f"\nSolution path for problem {i+1}:")
                solver.print_solution_path()
        
        return self._summarize_results(results)
    
    def _get_problem(self, dataset: Union[pd.DataFrame, Dict[str, Any], None], index: int) -> Dict[str, Any]:
        """Get problem from dataset"""
        if self.config.get('task') == 'game24':
            # Game24 task handles its own data loading
            if isinstance(dataset, dict):
                problem_ids = dataset.get('problem_ids', list(range(dataset.get('num_puzzles', 5))))
                problem_id = problem_ids[index] if index < len(problem_ids) else index
                
                # Get problem info from task
                problem_info = self.task.get_problem_info(problem_id)
                return {
                    'id': problem_id,
                    'numbers': [int(x) for x in problem_info['numbers'].split()]
                }
            else:
                return {'id': index}
                
        elif self.config.get('task') in ('math', 'gsm8k'):
            if isinstance(dataset, dict):
                # HuggingFace dataset metadata
                problem_ids = dataset.get('problem_ids', list(range(dataset.get('num_problems', 10))))
                problem_id = problem_ids[index] if index < len(problem_ids) else index

                # The task will load the actual problem
                if hasattr(self.task, 'get_problem_info'):
                    problem_info = self.task.get_problem_info(problem_id)
                    return {
                        'id': problem_id,
                        'problem': problem_info.get('problem_preview', ''),
                        'answer': problem_info.get('answer', '')
                    }
                else:
                    return {'id': problem_id}
            else:
                # Default case
                return {'id': index}
        else:
            raise ValueError(f"Unknown task: {self.config.get('task')}")
    
    
    def _summarize_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Summarize results"""
        successes = sum(1 for r in results if r['success'])
        total_time = sum(r['time'] for r in results)
        
        summary = {
            'success_rate': successes / len(results) if results else 0,
            'total_solved': successes,
            'total_problems': len(results),
            'avg_time': total_time / len(results) if results else 0,
            'total_time': total_time,
            'details': results
        }
        
        # Add strategy-specific metrics
        if results and 'nodes_explored' in results[0]:
            summary['avg_nodes_explored'] = sum(r.get('nodes_explored', 0) for r in results) / len(results)
            summary['avg_max_depth'] = sum(r.get('max_depth', 0) for r in results) / len(results)
        
        return summary
    
    def _analyze_comparison(self, results_by_strategy: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze comparison between strategies"""
        comparison = {
            'best_strategy': max(results_by_strategy.items(), 
                               key=lambda x: x[1]['success_rate'])[0],
            'strategy_ranking': sorted(results_by_strategy.items(), 
                                     key=lambda x: x[1]['success_rate'], 
                                     reverse=True)
        }
        
        if self.config.get('analytics', {}).get('show_comparison_matrix'):
            # Add detailed comparison matrix
            comparison['matrix'] = self._create_comparison_matrix(results_by_strategy)
        
        return comparison
    
    def _create_comparison_matrix(self, results_by_strategy: Dict[str, Dict]) -> pd.DataFrame:
        """Create comparison matrix"""
        strategies = list(results_by_strategy.keys())
        metrics = ['success_rate', 'avg_time', 'total_solved']
        
        data = []
        for strategy in strategies:
            row = {'strategy': strategy}
            for metric in metrics:
                row[metric] = results_by_strategy[strategy].get(metric, 0)
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _collect_all_details(self, results_by_strategy: Dict[str, Dict]) -> List[Dict]:
        """Collect all detailed results across strategies"""
        all_details = []
        for strategy_name, strategy_results in results_by_strategy.items():
            for detail in strategy_results.get('details', []):
                detail_copy = detail.copy()
                detail_copy['strategy'] = strategy_name
                all_details.append(detail_copy)
        return all_details
