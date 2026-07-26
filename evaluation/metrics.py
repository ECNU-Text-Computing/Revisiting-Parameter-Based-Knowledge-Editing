#!/usr/bin/env python
"""
Evaluation metrics for knowledge editing.

Implements the four core evaluation dimensions defined in the paper (Section 3):
  - Reliability: Does the edited model produce the updated target outputs?
  - Generalization: Does it generalize to paraphrased prompts?
  - Locality: Does it retain original behavior on unrelated queries?
  - Portability: Does it propagate edits to downstream reasoning tasks?

Usage:
    from evaluation.metrics import compute_all_metrics
    results = compute_all_metrics(edit_metrics, judge_results)
"""

from typing import Dict, List, Any, Optional
import numpy as np


def compute_reliability(judge_results: List[str]) -> float:
    """
    Reliability: fraction of edited prompts where the model produces the
    updated target output correctly.

    Eq. (2) in the paper.
    """
    if not judge_results:
        return 0.0
    correct = sum(1 for r in judge_results if r == "CORRECT")
    return correct / len(judge_results)


def compute_generalization(judge_results: List[str]) -> float:
    """
    Generalization: fraction of paraphrased/rephrased prompts where the
    model produces the correct updated answer.

    Eq. (3) in the paper.
    """
    if not judge_results:
        return 0.0
    correct = sum(1 for r in judge_results if r == "CORRECT")
    return correct / len(judge_results)


def compute_locality(
    pre_edit_answers: List[str],
    post_edit_answers: List[str],
    judge_fn=None,
) -> float:
    """
    Locality: fraction of unrelated queries where the edited model's output
    matches the original model's output.

    Eq. (4) in the paper. Uses exact match or judge-based consistency.
    """
    if not pre_edit_answers or not post_edit_answers:
        return 0.0
    if judge_fn is not None:
        # Judge-based locality: compare post-edit vs pre-edit answers
        matches = sum(
            1 for pre, post in zip(pre_edit_answers, post_edit_answers)
            if judge_fn(post, pre) == "CORRECT"
        )
    else:
        # Exact-match locality
        from .llm_judge import exact_match_score
        matches = sum(
            1 for pre, post in zip(pre_edit_answers, post_edit_answers)
            if exact_match_score(post, pre)
        )
    return matches / len(pre_edit_answers)


def compute_portability(judge_results: List[str]) -> float:
    """
    Portability: fraction of downstream reasoning prompts (aliases, causal,
    reverse relations) where the edited knowledge propagates correctly.

    Eq. (5) in the paper.
    """
    if not judge_results:
        return 0.0
    correct = sum(1 for r in judge_results if r == "CORRECT")
    return correct / len(judge_results)


def compute_all_metrics(
    reliability_results: Optional[List[str]] = None,
    generalization_results: Optional[List[str]] = None,
    pre_edit_locality: Optional[List[str]] = None,
    post_edit_locality: Optional[List[str]] = None,
    portability_results: Optional[List[str]] = None,
    judge_fn=None,
) -> Dict[str, float]:
    """
    Compute all four evaluation dimensions from judge results.

    Args:
        reliability_results: Judge verdicts for edited prompts.
        generalization_results: Judge verdicts for rephrased prompts.
        pre_edit_locality: Pre-edit answers for locality queries.
        post_edit_locality: Post-edit answers for locality queries.
        portability_results: Judge verdicts for portability prompts.
        judge_fn: Function for locality comparison (default: exact match).

    Returns:
        Dict with 'reliability', 'generalization', 'locality', 'portability'.
    """
    metrics = {}

    if reliability_results is not None:
        metrics['reliability'] = compute_reliability(reliability_results)
    if generalization_results is not None:
        metrics['generalization'] = compute_generalization(generalization_results)
    if pre_edit_locality is not None and post_edit_locality is not None:
        metrics['locality'] = compute_locality(
            pre_edit_locality, post_edit_locality, judge_fn
        )
    if portability_results is not None:
        metrics['portability'] = compute_portability(portability_results)

    return metrics


def summary_metrics(all_metrics: List[Dict]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple edits into mean scores.

    Args:
        all_metrics: List of per-edit metric dicts from BaseEditor.

    Returns:
        Dict with averaged metric scores.
    """
    agg = {
        'reliability': [],
        'generalization': [],
        'locality': [],
        'portability': [],
    }

    for m in all_metrics:
        post = m.get('post', {})
        for key in agg:
            val = post.get(key)
            if val is not None:
                try:
                    agg[key].append(float(val))
                except (ValueError, TypeError):
                    pass

    return {k: np.mean(v) if v else 0.0 for k, v in agg.items()}
