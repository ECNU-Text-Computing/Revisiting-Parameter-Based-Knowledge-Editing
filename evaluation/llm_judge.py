#!/usr/bin/env python
"""
LLM-as-a-Judge evaluation for knowledge editing.

Uses Qwen2.5-72B-Instruct (or another LLM) to judge semantic consistency between
predicted answers and gold targets. This is the primary evaluation metric used in
the paper (Section 5), as exact-match is too brittle for autoregressive decoding.

Usage:
    from evaluation.llm_judge import llm_judge, batch_llm_judge

    result = llm_judge("What city was Barack Obama born in?",
                       "Honolulu", "Obama was born in Honolulu, Hawaii.")
    # Returns "CORRECT"
"""

import re
import string
import time
from typing import List, Tuple, Optional

import regex
from openai import OpenAI


# ---------------------------------------------------------------------------
# Normalization (for exact match)
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """Normalize answer strings for comparison."""

    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check if prediction and ground truth match after normalization."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------
JUDGE_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT"].

The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: Malia and Sasha Obama are the names of Barack Obama's children.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.

The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Malia and Sasha, Malia and Sasha, Malia and Sasha, Malia and Sasha (repeated answer)
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target or contain repeated answer.

Here is a sample. Simply reply with either CORRECT or INCORRECT.

```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

According to the gold target, please grade the predicted answer of this question as one of:
CORRECT
INCORRECT

Only respond with one word: either CORRECT or INCORRECT."""


def llm_judge(
    question: str,
    ground_truth: str,
    prediction: str,
    client: Optional[OpenAI] = None,
    model: str = "Qwen2.5-72B-Instruct",
    api_key: str = "dummy",
    base_url: str = "http://localhost:21910/v1",
    max_retries: int = 3,
) -> str:
    """
    Judge whether a predicted answer is semantically consistent with the gold target.

    Args:
        question: The original question/prompt.
        ground_truth: The gold target answer.
        prediction: The model's predicted answer.
        client: OpenAI client (created if None).
        model: Judge model name.
        api_key: API key for the judge service.
        base_url: Base URL for the vLLM judge service.
        max_retries: Number of retries on failure.

    Returns:
        "CORRECT" or "INCORRECT".
    """
    if client is None:
        client = OpenAI(base_url=base_url, api_key=api_key)

    content = JUDGE_TEMPLATE.format(
        question=question,
        target=ground_truth,
        predicted_answer=prediction,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=10,
            )
            verdict = response.choices[0].message.content.strip().upper()
            if "CORRECT" in verdict:
                return "CORRECT"
            elif "INCORRECT" in verdict:
                return "INCORRECT"
            else:
                # Unexpected response — retry
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return "INCORRECT"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"LLM judge error after {max_retries} attempts: {e}")
            return "INCORRECT"

    return "INCORRECT"


def batch_llm_judge(
    questions: List[str],
    ground_truths: List[str],
    predictions: List[str],
    client: Optional[OpenAI] = None,
    **kwargs,
) -> List[str]:
    """
    Judge a batch of predictions.

    Returns a list of "CORRECT"/"INCORRECT" strings.
    """
    results = []
    for q, gt, pred in zip(questions, ground_truths, predictions):
        result = llm_judge(q, gt, pred, client=client, **kwargs)
        results.append(result)
    return results
