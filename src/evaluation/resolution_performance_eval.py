from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from collections import defaultdict

import json
import os

from .base import BaseMetric
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

logger = get_logger(__name__)


class ResolutionPerformanceMetric(BaseMetric):
    """
    A comprehensive metric that quantifies the relationship between resolution completeness 
    and performance throughout queries. This metric analyzes how the resolution status 
    of clues (fuzzy, traceable, resolved) correlates with QA performance.
    """
    
    metric_name: str = "resolution_performance"
    
    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)
    
    def calculate_metric_scores(self, 
                               query_results: List[Dict[str, Any]], 
                               gold_answers: List[List[str]], 
                               predicted_answers: List[str],
                               aggregation_fn: callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculate resolution-performance correlation metrics.
        
        Args:
            query_results: List of dictionaries containing resolution data for each query
            gold_answers: List of lists containing gold-standard answers for each query
            predicted_answers: List of predicted answers for each query
            aggregation_fn: Function to aggregate multiple gold answers (default: np.max)
            
        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A pooled dictionary with overall resolution-performance metrics
                - A list of dictionaries with per-query resolution-performance metrics
        """
        
        # First, calculate QA performance scores for each query
        qa_scores = self._calculate_qa_scores(gold_answers, predicted_answers, aggregation_fn)
        
        example_eval_results = []
        pooled_metrics = defaultdict(float)
        
        for i, (query_result, qa_score) in enumerate(zip(query_results, qa_scores)):
            # Extract resolution data from query result
            resolution_data = self._extract_resolution_data(query_result)
            
            if resolution_data is None:
                # Skip queries without resolution data (e.g., StandardRAG)
                example_eval_results.append({
                    "ResolutionCompleteness": 0.0,
                    "ResolutionEfficiency": 0.0,
                    "ResolutionPerformanceCorrelation": 0.0,
                    "CompletionStatus": 0.0,
                    "TerminationReason": "no_resolution_data",
                    "QA_ExactMatch": qa_score["ExactMatch"],
                    "QA_F1": qa_score["F1"]
                })
                continue
            
            # Calculate resolution metrics
            resolution_metrics = self._calculate_resolution_metrics(resolution_data)
            
            # Calculate resolution-performance correlation
            correlation_metrics = self._calculate_correlation_metrics(resolution_metrics, qa_score)
            
            # Combine all metrics for this query
            query_metrics = {
                **resolution_metrics,
                **correlation_metrics,
                "QA_ExactMatch": qa_score["ExactMatch"],
                "QA_F1": qa_score["F1"]
            }
            
            example_eval_results.append(query_metrics)
            
            # Accumulate for pooled results
            for key, value in query_metrics.items():
                if isinstance(value, (int, float)):
                    pooled_metrics[key] += value
        
        # Calculate averages for pooled results
        num_queries = len(query_results)
        for key in pooled_metrics:
            pooled_metrics[key] = round(pooled_metrics[key] / num_queries, 4)
        
        return dict(pooled_metrics), example_eval_results
    
    def _calculate_qa_scores(self, gold_answers: List[List[str]], predicted_answers: List[str], 
                           aggregation_fn: callable) -> List[Dict[str, float]]:
        """Calculate QA performance scores for each query using the centralized function."""
        from .qa_eval import calculate_qa_scores
        return calculate_qa_scores(gold_answers, predicted_answers, aggregation_fn)
    
    def _extract_resolution_data(self, query_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract resolution data from query result."""
        # Handle different result formats
        if isinstance(query_result, dict):
            # Check if this is already a complete resolution data structure
            if 'final_results' in query_result and 'step_1_reasoning' in query_result:
                return query_result
            
            # Direct dictionary format with completion_status at top level
            if 'completion_status' in query_result:
                return query_result
            elif 'metadata' in query_result and isinstance(query_result['metadata'], dict):
                return query_result['metadata']
        
        # Try to find resolution data in intermediate results
        if hasattr(query_result, 'question'):
            # QuerySolution object - try to find intermediate results
            query_id = f"query_{hash(query_result.question) % 10000}"
            intermediate_dir = getattr(self.global_config, 'save_dir', 'outputs')
            results_file = os.path.join(intermediate_dir, "intermediate_results", f"{query_id}_results.json")
            
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                    return data
                except:
                    pass
        
        return None
    
    def _calculate_resolution_metrics(self, resolution_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate resolution completeness and efficiency metrics."""
        metrics = {}
        
        # Extract final resolution counts
        final_results = resolution_data.get('final_results', {})
        step1_results = resolution_data.get('step_1_reasoning', {})
        
        # Get initial clue counts
        initial_fuzzy = len(step1_results.get('fuzzy_clues', []))
        initial_traceable = len(step1_results.get('traceable_clues', []))
        initial_resolved = len(step1_results.get('resolved_clues', []))
        total_initial = initial_fuzzy + initial_traceable + initial_resolved
        
        # Get final clue counts
        final_resolved = final_results.get('resolved_clues_count', 0)
        final_traceable = final_results.get('traceable_clues_count', 0)
        final_fuzzy = final_results.get('remaining_fuzzy_count', 0)
        total_final = final_resolved + final_traceable + final_fuzzy
        
        # Calculate resolution completeness (percentage of clues that became fully resolved)
        if total_initial > 0:
            resolution_completeness = final_resolved / total_initial
        else:
            resolution_completeness = 0.0
        
        # Calculate resolution efficiency (resolved per iteration)
        total_iterations = final_results.get('total_iterations', 1)
        if total_iterations > 0:
            resolution_efficiency = final_resolved / total_iterations
        else:
            resolution_efficiency = 0.0
        
        # Calculate completion status based on final_results completion_status
        # Only two possible values: "all_resolved" or "max_qa_steps_reached"
        completion_score = 0.0
        termination_reason = "unknown"
        
        completion_status = final_results.get('completion_status', 'unknown')
        if completion_status == 'all_resolved':
            completion_score = 1.0  # Fully resolved
            termination_reason = 'all_resolved'
        elif completion_status == 'max_qa_steps_reached':
            completion_score = 0.0  # Not fully resolved
            termination_reason = 'max_qa_steps_reached'
        else:
            completion_score = 0.0  # Unknown status
            termination_reason = 'unknown'
        
        # Calculate clue progression metrics
        iterations = resolution_data.get('iterations', [])
        if iterations:
            # Calculate average resolution rate per iteration
            total_resolved_across_iterations = sum(
                iter_data.get('step_3_clue_resolution', {}).get('resolved_count', 0) 
                for iter_data in iterations
            )
            avg_resolution_per_iteration = total_resolved_across_iterations / len(iterations) if iterations else 0
            
            # Calculate resolution momentum (acceleration of resolution)
            if len(iterations) >= 2:
                first_half = iterations[:len(iterations)//2]
                second_half = iterations[len(iterations)//2:]
                
                first_half_resolved = sum(
                    iter_data.get('step_3_clue_resolution', {}).get('resolved_count', 0) 
                    for iter_data in first_half
                )
                second_half_resolved = sum(
                    iter_data.get('step_3_clue_resolution', {}).get('resolved_count', 0) 
                    for iter_data in second_half
                )
                
                if first_half_resolved > 0:
                    resolution_momentum = second_half_resolved / first_half_resolved
                else:
                    resolution_momentum = 1.0 if second_half_resolved > 0 else 0.0
            else:
                resolution_momentum = 1.0
        else:
            avg_resolution_per_iteration = 0.0
            resolution_momentum = 1.0
        
        metrics.update({
            "ResolutionCompleteness": round(resolution_completeness, 4),
            "ResolutionEfficiency": round(resolution_efficiency, 4),
            "CompletionStatus": completion_score,
            "TerminationReason": termination_reason,
            "AvgResolutionPerIteration": round(avg_resolution_per_iteration, 4),
            "ResolutionMomentum": round(resolution_momentum, 4),
            "TotalIterations": total_iterations,
            "InitialClues": total_initial,
            "FinalResolvedClues": final_resolved,
            "FinalTraceableClues": final_traceable,
            "FinalFuzzyClues": final_fuzzy
        })
        
        return metrics
    
    def _calculate_correlation_metrics(self, resolution_metrics: Dict[str, float], 
                                     qa_score: Dict[str, float]) -> Dict[str, float]:
        """Calculate correlation between resolution metrics and QA performance."""
        metrics = {}
        
        # Resolution-Performance Correlation Score
        # This is a weighted combination that rewards both high resolution and high performance
        resolution_completeness = resolution_metrics.get("ResolutionCompleteness", 0.0)
        completion_status = resolution_metrics.get("CompletionStatus", 0.0)
        qa_em = qa_score.get("ExactMatch", 0.0)
        qa_f1 = qa_score.get("F1", 0.0)
        
        # Weighted correlation score
        correlation_score = (
            0.3 * resolution_completeness +
            0.2 * completion_status +
            0.25 * qa_em +
            0.25 * qa_f1
        )
        
        # Resolution-Performance Efficiency
        # How efficiently resolution translates to performance
        if resolution_completeness > 0:
            resolution_performance_efficiency = (qa_em + qa_f1) / 2 / resolution_completeness
        else:
            resolution_performance_efficiency = 0.0
        
        # Resolution-Performance Balance
        # How well resolution and performance are balanced
        resolution_performance_balance = 1.0 - abs(resolution_completeness - (qa_em + qa_f1) / 2)
        
        metrics.update({
            "ResolutionPerformanceCorrelation": round(correlation_score, 4),
            "ResolutionPerformanceEfficiency": round(resolution_performance_efficiency, 4),
            "ResolutionPerformanceBalance": round(resolution_performance_balance, 4)
        })
        
        return metrics
    
    def calculate_overall_insights(self, pooled_metrics: Dict[str, float], 
                                 example_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate overall insights about resolution-performance relationship."""
        insights = {}
        
        # Categorize queries by termination reason
        all_resolved_queries = [m for m in example_metrics if m.get("TerminationReason") == "all_resolved"]
        max_steps_queries = [m for m in example_metrics if m.get("TerminationReason") == "max_qa_steps_reached"]
        no_data_queries = [m for m in example_metrics if m.get("TerminationReason") == "no_resolution_data"]
        unknown_queries = [m for m in example_metrics if m.get("TerminationReason") not in ["all_resolved", "max_qa_steps_reached", "no_resolution_data"]]
        
        # All resolved queries insights
        if all_resolved_queries:
            insights["num_all_resolved"] = len(all_resolved_queries)
            insights["avg_qa_em_all_resolved"] = np.mean([m["QA_ExactMatch"] for m in all_resolved_queries])
            insights["avg_qa_f1_all_resolved"] = np.mean([m["QA_F1"] for m in all_resolved_queries])
            insights["avg_resolution_completeness_all_resolved"] = np.mean([m["ResolutionCompleteness"] for m in all_resolved_queries])
            insights["avg_resolution_efficiency_all_resolved"] = np.mean([m["ResolutionEfficiency"] for m in all_resolved_queries])
            insights["avg_iterations_all_resolved"] = np.mean([m["TotalIterations"] for m in all_resolved_queries])
        
        # Max steps reached queries insights
        if max_steps_queries:
            insights["num_max_steps_reached"] = len(max_steps_queries)
            insights["avg_qa_em_max_steps"] = np.mean([m["QA_ExactMatch"] for m in max_steps_queries])
            insights["avg_qa_f1_max_steps"] = np.mean([m["QA_F1"] for m in max_steps_queries])
            insights["avg_resolution_completeness_max_steps"] = np.mean([m["ResolutionCompleteness"] for m in max_steps_queries])
            insights["avg_resolution_efficiency_max_steps"] = np.mean([m["ResolutionEfficiency"] for m in max_steps_queries])
            insights["avg_iterations_max_steps"] = np.mean([m["TotalIterations"] for m in max_steps_queries])
        
        # No resolution data queries insights
        if no_data_queries:
            insights["num_no_resolution_data"] = len(no_data_queries)
            insights["avg_qa_em_no_data"] = np.mean([m["QA_ExactMatch"] for m in no_data_queries])
            insights["avg_qa_f1_no_data"] = np.mean([m["QA_F1"] for m in no_data_queries])
        
        # Unknown queries insights
        if unknown_queries:
            insights["num_unknown_status"] = len(unknown_queries)
            insights["avg_qa_em_unknown"] = np.mean([m["QA_ExactMatch"] for m in unknown_queries])
            insights["avg_qa_f1_unknown"] = np.mean([m["QA_F1"] for m in unknown_queries])
        
        # Resolution efficiency insights
        resolution_completeness_scores = [m["ResolutionCompleteness"] for m in example_metrics if m["ResolutionCompleteness"] > 0]
        if resolution_completeness_scores:
            insights["avg_resolution_completeness"] = np.mean(resolution_completeness_scores)
            insights["std_resolution_completeness"] = np.std(resolution_completeness_scores)
        
        # Correlation insights
        correlation_scores = [m["ResolutionPerformanceCorrelation"] for m in example_metrics if m["ResolutionPerformanceCorrelation"] > 0]
        if correlation_scores:
            insights["avg_correlation"] = np.mean(correlation_scores)
            insights["std_correlation"] = np.std(correlation_scores)
        
        # Resolution rate (binary - only all_resolved vs others)
        total_queries_with_data = len(all_resolved_queries) + len(max_steps_queries)
        if total_queries_with_data > 0:
            insights["resolution_rate"] = len(all_resolved_queries) / total_queries_with_data
            insights["max_steps_rate"] = len(max_steps_queries) / total_queries_with_data
        
        # Overall statistics
        total_queries = len(example_metrics)
        insights["total_queries"] = total_queries
        insights["all_resolved_percentage"] = len(all_resolved_queries) / total_queries * 100 if total_queries > 0 else 0
        insights["max_steps_percentage"] = len(max_steps_queries) / total_queries * 100 if total_queries > 0 else 0
        insights["no_data_percentage"] = len(no_data_queries) / total_queries * 100 if total_queries > 0 else 0
        
        return insights 