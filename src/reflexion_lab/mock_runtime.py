from __future__ import annotations
import os
import re
import json
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .llm import call_llm

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}

def is_mock_mode() -> bool:
    return os.environ.get("MOCK_MODE", "false").lower() == "true"

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> tuple[str, int, int]:
    if is_mock_mode():
        # Mock mode logic
        if example.qid not in FIRST_ATTEMPT_WRONG:
            ans = example.gold_answer
        elif agent_type == "react":
            ans = FIRST_ATTEMPT_WRONG[example.qid]
        elif attempt_id == 1 and not reflection_memory:
            ans = FIRST_ATTEMPT_WRONG[example.qid]
        else:
            ans = example.gold_answer
        return ans, 150 + attempt_id * 50, 100 + attempt_id * 20

    # Real LLM logic
    formatted_context = "\n\n".join(f"Title: {chunk.title}\n{chunk.text}" for chunk in example.context)
    formatted_reflections = "\n".join(reflection_memory) if reflection_memory else "No previous attempts."
    
    user_prompt = f"""Context:
{formatted_context}

Question: {example.question}

[Reflection Memory]
{formatted_reflections}

Answer:"""
    
    res = call_llm(system_prompt=ACTOR_SYSTEM, user_prompt=user_prompt, max_tokens=256)
    return res["text"].strip(), res["input_tokens"] + res["output_tokens"], res["latency_ms"]

def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, int, int]:
    if is_mock_mode():
        # Mock mode logic
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            judge = JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
        elif normalize_answer(answer) == "london":
            judge = JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
        else:
            judge = JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])
        return judge, 120, 80

    # Real LLM logic
    user_prompt = f"""Question: {example.question}
Gold Answer: {example.gold_answer}
Predicted Answer: {answer}"""

    res = call_llm(system_prompt=EVALUATOR_SYSTEM, user_prompt=user_prompt, max_tokens=512, json_mode=True)
    text = res["text"]
    
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            parsed = json.loads(text)
        judge = JudgeResult(
            score=int(parsed.get("score", 0)),
            reason=str(parsed.get("reason", "Failed to parse evaluation")),
            missing_evidence=list(parsed.get("missing_evidence", [])),
            spurious_claims=list(parsed.get("spurious_claims", []))
        )
    except Exception as e:
        is_correct = normalize_answer(example.gold_answer) == normalize_answer(answer)
        judge = JudgeResult(
            score=1 if is_correct else 0,
            reason=f"Failed to parse LLM evaluation (Regex/JSON error: {str(e)}). Fallback: {is_correct}",
            missing_evidence=[],
            spurious_claims=[]
        )
        
    return judge, res["input_tokens"] + res["output_tokens"], res["latency_ms"]

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> tuple[ReflectionEntry, int, int]:
    if is_mock_mode():
        # Mock mode logic
        strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
        reflection = ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)
        return reflection, 100, 60

    # Real LLM logic
    formatted_context = "\n\n".join(f"Title: {chunk.title}\n{chunk.text}" for chunk in example.context)
    user_prompt = f"""Context:
{formatted_context}

Question: {example.question}
Failed Answer: {judge.reason} (Evaluator failure description)

Please provide reflection on why the answer failed, and suggest a strategy for the next attempt."""

    res = call_llm(system_prompt=REFLECTOR_SYSTEM, user_prompt=user_prompt, max_tokens=512, json_mode=True)
    text = res["text"]
    
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            parsed = json.loads(text)
        reflection = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson=str(parsed.get("lesson", "Unknown error")),
            next_strategy=str(parsed.get("next_strategy", "Try verifying the facts again."))
        )
    except Exception as e:
        reflection = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson=f"Failed to parse reflection (Regex/JSON error: {str(e)})",
            next_strategy="Analyze the question and context carefully and double-check each logic step."
        )
        
    return reflection, res["input_tokens"] + res["output_tokens"], res["latency_ms"]
