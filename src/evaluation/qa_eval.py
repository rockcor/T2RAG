from typing import List, Dict, Tuple, Optional, Union, Callable
from collections import Counter
import numpy as np

from .base import BaseMetric
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig
from ..utils.eval_utils import normalize_answer

logger = get_logger(__name__)

CANNOT_ANSWER_VARIATIONS = [
    "cannot answer",
    "can't answer",
    "no answer",
    "unanswerable",
    "insufficient information",
    "not enough information",
    "cannot be answered",
    "can't be answered",
    "impossible to answer",
    "no sufficient information",
    "lack of information",
    "missing information",
    "incomplete information",
    "not provided",
    "not available",
    "unknown",
    "undetermined",
    "cannot determine",
    "can't determine",
    "cannot find",
    "can't find",
    "not found",
    "not specified",
    "not stated",
    "not given",
    "not mentioned"
]

def is_cannot_answer(answer: str) -> bool:
    """
    Check if an answer indicates inability to answer the question.
    
    Args:
        answer: The answer string to check
        
    Returns:
        bool: True if the answer indicates inability to answer
    """
    normalized = normalize_answer(answer).lower()
    return any(variation in normalized for variation in CANNOT_ANSWER_VARIATIONS)
# Reference: MRQA official eval
class QAExactMatch(BaseMetric):
    metric_name: str = "qa_exact_match"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)

    def calculate_metric_scores(self, gold_answers: Union[List[List[str]], List[str], List[Union[str, List[str]]]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates the Exact Match (EM) score, handling pipe-separated gold answers.
        Supports PopQA format with pipe-separated gold answers.
        """
        # Normalize and expand gold answers
        processed_gold = []
        if isinstance(gold_answers, list):
            for item in gold_answers:
                # Ensure the item is a list of strings
                temp_list = [item] if isinstance(item, str) else [str(g) for g in item] if isinstance(item, list) else [str(item)]
                
                # Expand the list by splitting strings containing '|'
                expanded_list = []
                for gold_str in temp_list:
                    expanded_list.extend([ans.strip() for ans in gold_str.split('|') if ans.strip()])
                
                # Remove duplicates while preserving order
                seen = set()
                unique_answers = []
                for ans in expanded_list:
                    if ans not in seen:
                        seen.add(ans)
                        unique_answers.append(ans)
                
                processed_gold.append(unique_answers)
        
        gold_answers = processed_gold
        
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

        example_eval_results = []
        total_em = 0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            em_scores = []
            # Special case for "insufficient information"
            is_no_answer_match = any(
                normalize_answer(gold).lower() == "insufficient information" and is_cannot_answer(predicted)
                for gold in gold_list
            )
            
            if is_no_answer_match:
                em_scores.append(1.0)
            else:
                for gold in gold_list:
                    score = 1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0
                    em_scores.append(score)
            
            aggregated_em = aggregation_fn(em_scores) if em_scores else 0.0
            example_eval_results.append({"ExactMatch": aggregated_em})
            total_em += aggregated_em

        avg_em = total_em / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"ExactMatch": avg_em}

        return pooled_eval_results, example_eval_results
    
class QAF1Score(BaseMetric):
    metric_name: str = "qa_f1_score"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)

    def calculate_metric_scores(self, gold_answers: Union[List[List[str]], List[str], List[Union[str, List[str]]]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates the F1 score, handling pipe-separated gold answers.
        Supports PopQA format with pipe-separated gold answers.
        """
        # Normalize and expand gold answers
        processed_gold = []
        if isinstance(gold_answers, list):
            for item in gold_answers:
                # Ensure the item is a list of strings
                temp_list = [item] if isinstance(item, str) else [str(g) for g in item] if isinstance(item, list) else [str(item)]
                
                # Expand the list by splitting strings containing '|'
                expanded_list = []
                for gold_str in temp_list:
                    expanded_list.extend([ans.strip() for ans in gold_str.split('|') if ans.strip()])
                
                # Remove duplicates while preserving order
                seen = set()
                unique_answers = []
                for ans in expanded_list:
                    if ans not in seen:
                        seen.add(ans)
                        unique_answers.append(ans)
                
                processed_gold.append(unique_answers)
        
        gold_answers = processed_gold

        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

        def compute_f1(gold: str, predicted: str) -> float:
            gold_tokens = normalize_answer(gold).split()
            predicted_tokens = normalize_answer(predicted).split()
            
            if not gold_tokens or not predicted_tokens:
                return 0.0

            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = 1.0 * num_same / len(predicted_tokens)
            recall = 1.0 * num_same / len(gold_tokens)
            return (2 * precision * recall) / (precision + recall)

        example_eval_results = []
        total_f1 = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            f1_scores = []
            # Special case for "insufficient information"
            is_no_answer_match = any(
                normalize_answer(gold).lower() == "insufficient information" and is_cannot_answer(predicted)
                for gold in gold_list
            )

            if is_no_answer_match:
                f1_scores.append(1.0)
            else:
                for gold in gold_list:
                    f1_scores.append(compute_f1(gold, predicted))
            
            aggregated_f1 = aggregation_fn(f1_scores) if f1_scores else 0.0
            example_eval_results.append({"F1": aggregated_f1})
            total_f1 += aggregated_f1

        avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"F1": avg_f1}

        return pooled_eval_results, example_eval_results


def calculate_qa_scores(gold_answers: List[List[str]], predicted_answers: List[str], 
                       aggregation_fn: callable = np.max) -> List[Dict[str, float]]:
    """
    Centralized function to calculate QA performance scores for each query.
    This function can be used by all methods to ensure consistent evaluation.
    Supports PopQA format with pipe-separated gold answers.
    
    Args:
        gold_answers: List of lists containing gold-standard answers for each query
        predicted_answers: List of predicted answers for each query
        aggregation_fn: Function to aggregate multiple gold answers (default: np.max)
        
    Returns:
        List[Dict[str, float]]: List of dictionaries with "ExactMatch" and "F1" scores for each query
    """
    # Calculate per-query scores
    qa_scores = []
    for gold_list, predicted in zip(gold_answers, predicted_answers):
        # Expand gold answers to handle pipe-separated format (PopQA compatibility)
        expanded_gold_answers = []
        for gold in gold_list:
            if isinstance(gold, str):
                # Split on '|' to handle pipe-separated answers
                expanded_gold_answers.extend([ans.strip() for ans in gold.split('|') if ans.strip()])
            else:
                expanded_gold_answers.append(str(gold))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_gold_answers = []
        for ans in expanded_gold_answers:
            if ans not in seen:
                seen.add(ans)
                unique_gold_answers.append(ans)
        
        # Calculate EM score
        em_scores = []
        # Special case for "insufficient information"
        is_no_answer_match = any(
            normalize_answer(gold).lower() == "insufficient information" and is_cannot_answer(predicted)
            for gold in unique_gold_answers
        )
        
        if is_no_answer_match:
            em_scores.append(1.0)
        else:
            for gold in unique_gold_answers:
                score = 1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0
                em_scores.append(score)
        
        em_score = aggregation_fn(em_scores) if em_scores else 0.0
        
        # Calculate F1 score
        f1_scores = []
        if is_no_answer_match:
            f1_scores.append(1.0)
        else:
            for gold in unique_gold_answers:
                gold_tokens = normalize_answer(gold).split()
                predicted_tokens = normalize_answer(predicted).split()
                
                if not gold_tokens or not predicted_tokens:
                    f1_scores.append(0.0)
                    continue
                
                common = Counter(predicted_tokens) & Counter(gold_tokens)
                num_same = sum(common.values())
                
                if num_same == 0:
                    f1_scores.append(0.0)
                    continue
                
                precision = 1.0 * num_same / len(predicted_tokens)
                recall = 1.0 * num_same / len(gold_tokens)
                f1 = (2 * precision * recall) / (precision + recall)
                f1_scores.append(f1)
        
        f1_score = aggregation_fn(f1_scores) if f1_scores else 0.0
        
        qa_scores.append({
            "ExactMatch": em_score,
            "F1": f1_score
        })
    
    return qa_scores


def calculate_overall_qa_metrics(gold_answers: List[List[str]], predicted_answers: List[str], 
                                aggregation_fn: callable = np.max) -> Dict[str, float]:
    """
    Calculate overall QA metrics (averages across all queries).
    
    Args:
        gold_answers: List of lists containing gold-standard answers for each query
        predicted_answers: List of predicted answers for each query
        aggregation_fn: Function to aggregate multiple gold answers (default: np.max)
        
    Returns:
        Dict[str, float]: Dictionary with overall "ExactMatch" and "F1" scores
    """
    qa_scores = calculate_qa_scores(gold_answers, predicted_answers, aggregation_fn)
    
    # Calculate overall averages
    overall_em = np.mean([score["ExactMatch"] for score in qa_scores])
    overall_f1 = np.mean([score["F1"] for score in qa_scores])
    
    return {
        'ExactMatch': round(overall_em, 4),
        'F1': round(overall_f1, 4)
    }