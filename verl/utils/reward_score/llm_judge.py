# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LLM-as-Judge reward score: call a remote LLM (OpenAI-compatible, e.g. vLLM) to judge
whether model response is correct. Uses TONGYI prompt (A/B or [CORRECT]/[INCORRECT]).

Supports: async single/batch, rate limiting via semaphore, retries with backoff.
Multiple judge vLLM instances: pass endpoint_list (e.g. two ports) to spread requests.
"""

import asyncio
import json
import logging
import os
import random
import time
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Judge prompt (TONGYI: A/B or [CORRECT]/[INCORRECT])
# ---------------------------------------------------------------------------

LLM_AS_JUDGE_TONGYI_PROMPT = """
Based on the given question, standard answer, and model-predicted answer, evaluate whether the model's response is correct. Your task is to classify the result as: [CORRECT] or [INCORRECT].

First, we'll list examples for each category, then you'll evaluate a new question's predicted answer.
Here are examples of [CORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia Obama and Sasha Obama
Model Prediction 2: Malia and Sasha
Model Prediction 3: Most would say Malia and Sasha, but I'm not sure, I should verify
Model Prediction 4: Barack Obama has two daughters, Malia Ann and Natasha Marian, commonly known as Malia Obama and Sasha Obama.
```
These responses are all [CORRECT] because they:
    - Fully include the important information from the standard answer.
    - Don't contain any information that contradicts the standard answer.
    - Focus only on semantic content; language, capitalization, punctuation, grammar, and order aren't important.
    - Vague statements or guesses are acceptable as long as they include the standard answer and don't contain incorrect information or contradictions.

Here are examples of [INCORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia
Model Prediction 2: Malia, Sasha and Susan or Sasha Obama or Malia Obama, or Natasha Marian, or Einstein
Model Prediction 3: While I don't know their exact names, I can tell you Barack Obama has two children.
Model Prediction 4: You might be thinking of Betsy and Olivia. But you should verify the details with the latest references. Is that the correct answer?
Model Prediction 5: Barack Obama's children
```
These responses are all [INCORRECT] because they:
    - Contain factual statements that contradict the standard answer.
    - Are empty or merely repeat the question.
    - Enumerate multiple answers or repeat the answer.

Pay special attention to the following:
- The standard answer may contain responses to multiple aspects of the question, and within the same aspect, there might be different descriptions, all of which are correct and are given in the same bracket, connected by commas. For example, for the question "What is the name of ByteDance's AI model?", the standard answer is "[[Doubao, Skylark]]":
    - Predicted answers "Doubao", "Doubao, Skylark", "Skylark", etc. are all [CORRECT].
- For standard answers containing responses to different aspects, the model needs to provide answers to all aspects to be considered correct; otherwise, it's directly judged as [INCORRECT]. There is no [PARTIALLY CORRECT] output option. These answers will be given in different brackets. For example, for the question "Who are the members of TFBOYS?", the standard answer is "[[Wang Junkai][Wang Yuan][Yi Yangqianxi]]":
    - Predicted answers like "Wang Junkai, Wang Yuan, Yi Yangqianxi" that include all answers are [CORRECT].
    - Predicted answers like "Wang Junkai, Yi Yangqianxi" that don't include all answers are [INCORRECT].

Also note the following points:
- For questions with numerical standard answers, the predicted answer should match the standard answer. For example, for the question "What is the total length in meters of the Huangpu River Bridge on the Jinshan Railway?", the standard answer is "3518.17":
    - Predicted answers "3518", "3518.1", "3518.17" are all [CORRECT].
    - Predicted answers "3520" and "3600" are [INCORRECT].
- If the model prediction doesn't directly answer the question, attempts to circumvent or fails to directly provide the standard answer, it's considered an [INCORRECT] answer.
    - For example, for the question "Who is JJ Lin's wife?", with the standard answer "Ding Wenqi", model predictions like "JJ Lin's wife", "JJ Lin's wife should be excellent", "JJ Lin's wife might be a public figure" are all [INCORRECT].
- If the standard answer contains more information than the question asks for, the predicted answer only needs to include the information mentioned in the question.
    - For example, for the question "What is the main chemical component of magnesite?", with the standard answer "Magnesium carbonate (MgCO3)", "Magnesium carbonate" or "MgCO3" are both considered [CORRECT] answers.
- If information omitted in the predicted answer can be clearly inferred from the question, it's considered correct.
    - For example, for the question "The Nuragic ruins of Barumini were listed as a World Cultural Heritage by UNESCO in 1997, so where is this site located?", with the standard answer "Sardinia, Italy", the predicted answer "Sardinia" is considered [CORRECT].
- If it's clear that different translations of a name refer to the same person, it's considered correct.
    - For example, if the standard answer is "Robinson", answers like "Lubinson" or "Lubinsun" are both correct.
- You should focus more on the match between the standard answer and the model prediction, rather than whether the standard answer itself is correct.

Below is a new question example. Please reply with only [CORRECT] or [INCORRECT], without apologies or corrections to your own errors, just evaluate the answer.
```
Question: {question}
Standard Answer: {correct_answer}
Predicted Answer: {response}
```

Evaluate this new question's predicted answer as one of the following:
A. [CORRECT]
B. [INCORRECT]

Return only the option representing [CORRECT] or [INCORRECT], i.e. just return A or B, without adding any other text.
""".strip()


def _parse_judge_output(output: str) -> dict[str, Any] | str:
    """Parse TONGYI judge output: A/B or [CORRECT]/[INCORRECT]. Returns dict or 'PARSE FAILED'."""
    if not output or not isinstance(output, str):
        return "PARSE FAILED"
    raw = output.strip()
    first_line = raw.split("\n")[0].strip() if raw else ""
    upper = raw.upper()
    if first_line.startswith("A") or first_line.startswith("A."):
        return {"correct": "yes", "accuracy": 1.0}
    if first_line.startswith("B") or first_line.startswith("B."):
        return {"correct": "no", "accuracy": 0.0}
    if "[CORRECT]" in upper and "[INCORRECT]" not in upper:
        return {"correct": "yes", "accuracy": 1.0}
    if "[INCORRECT]" in upper:
        return {"correct": "no", "accuracy": 0.0}
    return "PARSE FAILED"


def _default_judge_result() -> dict[str, Any]:
    return {"correct": "no", "accuracy": 0.0}


def _log_judge_sample(
    data_source: str,
    extra_info: dict,
    question: str,
    solution_str: str,
    ground_truth: str,
    result: dict[str, Any],
) -> None:
    """Log one judge sample for debugging. Supports multi-turn prompt messages."""
    split = extra_info.get("split", "unknown")
    logger.info("================llm_judge %s data sample================", split)
    logger.info("data_source: %s", data_source)

    prompt_messages = extra_info.get("prompt_messages")
    if prompt_messages and isinstance(prompt_messages, list) and len(prompt_messages) > 2:
        k_selected = extra_info.get("k_selected", "?")
        n_total = extra_info.get("n_total_rounds", "?")
        logger.info("----------------prompt messages (%s/%s rounds)----------------", k_selected, n_total)
        for msg in prompt_messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "unknown")
            content = (msg.get("content") or "").strip()
            logger.info("----[%s]----", role)
            # truncate very long content (system prompt, tool_response) for readability
            if len(content) > 500:
                logger.info("%s ... (truncated, total %d chars)", content[:500], len(content))
            else:
                for line in content.splitlines():
                    logger.info(line)
    else:
        logger.info("----------------question----------------")
        for line in (question or "").strip().splitlines():
            logger.info(line)

    logger.info("----------------solution str----------------")
    for line in solution_str.splitlines():
        logger.info(line)
    logger.info("----------------ground truth----------------")
    logger.info(ground_truth)
    logger.info("correct: %s, accuracy: %s, score: %s",
                result.get("correct"), result.get("accuracy"), result.get("score"))
    logger.info("================end================")


def _normalize_endpoints(endpoint: str | None = None, endpoint_list: list[str] | None = None) -> list[str]:
    """Return list of endpoints. Prefer endpoint_list if given; else [endpoint]. At least one required."""
    if endpoint_list and len(endpoint_list) > 0:
        return list(endpoint_list)
    if endpoint:
        return [endpoint]
    raise ValueError("llm_judge: provide either 'endpoint' or 'endpoint_list' in reward_kwargs.")


# One client per endpoint to reuse connections when spreading across multiple vLLM instances
_client_cache: dict[tuple[str, str], Any] = {}


def _get_client(endpoint: str, api_key: str) -> Any:
    """Get or create AsyncOpenAI client for endpoint (cached)."""
    key = (endpoint, api_key)
    if key not in _client_cache:
        from openai import AsyncOpenAI
        _client_cache[key] = AsyncOpenAI(base_url=endpoint, api_key=api_key)
    return _client_cache[key]


# ---------------------------------------------------------------------------
# Endpoint health management: failover & temporary blocking
# ---------------------------------------------------------------------------

class EndpointManager:
    """Tracks endpoint health, blocks unreachable ones temporarily, and provides failover.

    - After ``block_threshold`` consecutive failures an endpoint is blocked for
      ``cooldown_seconds``.  Blocked endpoints are skipped when choosing where to
      send the next request.
    - When all endpoints are blocked the manager resets so that every endpoint
      is eligible again (avoids permanent deadlock).
    - Endpoints are automatically unblocked once the cooldown expires.
    """

    def __init__(
        self,
        endpoints: list[str],
        block_threshold: int = 3,
        cooldown_seconds: float = 120.0,
    ):
        self._endpoints = list(endpoints)
        self._block_threshold = block_threshold
        self._cooldown_seconds = cooldown_seconds
        self._failure_counts: dict[str, int] = {ep: 0 for ep in endpoints}
        self._blocked_until: dict[str, float] = {}

    def get_endpoint(self) -> str:
        """Pick a random healthy endpoint.  Unblocks expired ones first."""
        now = time.monotonic()
        expired = [ep for ep, until in self._blocked_until.items() if now >= until]
        for ep in expired:
            del self._blocked_until[ep]
            self._failure_counts[ep] = 0
            logger.info("llm_judge: endpoint %s unblocked after cooldown", ep)

        healthy = [ep for ep in self._endpoints if ep not in self._blocked_until]
        if not healthy:
            logger.warning("llm_judge: all endpoints blocked — resetting to retry all")
            self._blocked_until.clear()
            self._failure_counts = {ep: 0 for ep in self._endpoints}
            healthy = list(self._endpoints)

        return random.choice(healthy)

    def report_success(self, endpoint: str) -> None:
        self._failure_counts[endpoint] = 0
        self._blocked_until.pop(endpoint, None)

    def report_failure(self, endpoint: str) -> None:
        self._failure_counts[endpoint] = self._failure_counts.get(endpoint, 0) + 1
        if (
            self._failure_counts[endpoint] >= self._block_threshold
            and endpoint not in self._blocked_until
        ):
            self._blocked_until[endpoint] = time.monotonic() + self._cooldown_seconds
            logger.warning(
                "llm_judge: endpoint %s blocked after %d consecutive failures (cooldown %.0fs)",
                endpoint,
                self._failure_counts[endpoint],
                self._cooldown_seconds,
            )

    @property
    def status(self) -> str:
        healthy = [ep for ep in self._endpoints if ep not in self._blocked_until]
        blocked = list(self._blocked_until.keys())
        return f"healthy={healthy}, blocked={blocked}"


_endpoint_manager_cache: dict[tuple[str, ...], EndpointManager] = {}


def _get_endpoint_manager(endpoints: list[str]) -> EndpointManager:
    """Get or create an EndpointManager for a given set of endpoints (cached)."""
    key = tuple(sorted(endpoints))
    if key not in _endpoint_manager_cache:
        _endpoint_manager_cache[key] = EndpointManager(endpoints)
    return _endpoint_manager_cache[key]


async def _judge_single_async(
    question: str,
    response: str,
    correct_answer: str,
    endpoint_mgr: EndpointManager,
    model_name: str,
    api_key: str | None = None,
    max_retries: int = 5,
    timeout: float = 60.0,
    semaphore: asyncio.Semaphore | None = None,
) -> dict[str, Any]:
    """
    Call remote LLM (OpenAI-compatible) to judge (question, response, correct_answer).
    Switches to a different healthy endpoint on each retry via *endpoint_mgr*.
    Optional semaphore for rate limiting. Returns dict with 'accuracy' (0.0 or 1.0).
    """
    try:
        from openai import AsyncOpenAI  # noqa: F401
    except ImportError as e:
        raise ImportError("llm_judge requires openai package: pip install openai") from e

    api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    prompt = LLM_AS_JUDGE_TONGYI_PROMPT.format(question=question, response=response, correct_answer=correct_answer)
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries):
        ep = endpoint_mgr.get_endpoint()
        client = _get_client(ep, api_key)

        async def _request(_client=client) -> str:
            resp = await _client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()

        try:
            if semaphore:
                async with semaphore:
                    output = await asyncio.wait_for(_request(), timeout=timeout)
            else:
                output = await asyncio.wait_for(_request(), timeout=timeout)
            endpoint_mgr.report_success(ep)
            result = _parse_judge_output(output)
            if result != "PARSE FAILED":
                return result
            logger.warning(
                "llm_judge parse failed attempt %s/%s (endpoint %s), retrying...",
                attempt + 1,
                max_retries,
                ep,
            )
        except asyncio.TimeoutError:
            endpoint_mgr.report_failure(ep)
            logger.warning("llm_judge timeout attempt %s/%s (endpoint %s)", attempt + 1, max_retries, ep)
        except Exception as e:
            endpoint_mgr.report_failure(ep)
            logger.warning("llm_judge attempt %s/%s error (endpoint %s): %s", attempt + 1, max_retries, ep, e)
        if attempt < max_retries - 1:
            jitter = random.uniform(0.8, 1.2)
            await asyncio.sleep(min((2 ** attempt) * jitter, 10.0))
    return _default_judge_result()


# ---------------------------------------------------------------------------
# Public API for verl reward_manager
# ---------------------------------------------------------------------------

async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    *,
    endpoint: str | None = None,
    endpoint_list: list[str] | None = None,
    model_name: str = "",
    api_key: str | None = None,
    max_concurrent: int = 32,
    max_retries: int = 5,
    timeout: float = 60.0,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Async single-sample LLM judge. Used by NaiveRewardManager when custom_reward_function
    points to this module with name='compute_score'.

    extra_info should contain 'question' (the prompt text for the judge).
    Pass either endpoint (single URL) or endpoint_list (multiple URLs); requests are
    spread randomly across endpoint_list to reduce load per vLLM instance.
    Returns dict with 'score' and 'accuracy' (0.0/1.0).
    """
    extra_info = extra_info or {}
    question = extra_info.get("question") or extra_info.get("prompt") or ""
    endpoints = _normalize_endpoints(endpoint=endpoint, endpoint_list=endpoint_list)
    mgr = _get_endpoint_manager(endpoints)
    sem = _get_semaphore_for_loop(endpoints, max_concurrent)
    result = await _judge_single_async(
        question=question,
        response=solution_str,
        correct_answer=ground_truth,
        endpoint_mgr=mgr,
        model_name=model_name,
        api_key=api_key,
        max_retries=max_retries,
        timeout=timeout,
        semaphore=sem,
    )
    score = result["accuracy"]
    out = {"score": score, "accuracy": score, **result} if isinstance(score, (int, float)) else result

    do_print = random.randint(1, 16) == 1
    if do_print:
        _log_judge_sample(
            data_source=data_source,
            extra_info=extra_info,
            question=question,
            solution_str=solution_str,
            ground_truth=ground_truth,
            result=out,
        )
    return out


_semaphore_cache: dict[tuple[tuple[str, ...], int, int], asyncio.Semaphore] = {}


def _get_semaphore_for_loop(endpoints: list[str], max_concurrent: int) -> asyncio.Semaphore:
    """Get or create a semaphore for this endpoint (or pool) in the current event loop."""
    try:
        loop = asyncio.get_running_loop()
        key = (tuple(endpoints), max_concurrent, id(loop))
        if key not in _semaphore_cache:
            _semaphore_cache[key] = asyncio.Semaphore(max_concurrent)
        return _semaphore_cache[key]
    except RuntimeError:
        return asyncio.Semaphore(max_concurrent)


def compute_score_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict],
    *,
    endpoint: str | None = None,
    endpoint_list: list[str] | None = None,
    model_name: str = "",
    api_key: str | None = None,
    max_concurrent: int = 32,
    max_retries: int = 5,
    timeout: float = 60.0,
    **kwargs: Any,
) -> list[float | dict[str, Any]]:
    """
    Sync batch LLM judge. Used by BatchRewardManager when custom_reward_function
    points to this module with name='compute_score_batch' and reward_manager='batch'.

    Pass either endpoint or endpoint_list; each request picks a random endpoint from
    endpoint_list to spread load across multiple vLLM instances.
    Returns list of dict with 'score' and 'accuracy'.
    """
    endpoints = _normalize_endpoints(endpoint=endpoint, endpoint_list=endpoint_list)
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    sem = asyncio.Semaphore(max_concurrent)

    async def _batch() -> list[float | dict[str, Any]]:
        try:
            from openai import AsyncOpenAI  # noqa: F401
        except ImportError as e:
            raise ImportError("llm_judge requires openai package: pip install openai") from e

        mgr = _get_endpoint_manager(endpoints)

        async def _one(i: int) -> float | dict[str, Any]:
            ex = extra_infos[i] if i < len(extra_infos) else {}
            question = ex.get("question") or ex.get("prompt") or ""
            prompt = LLM_AS_JUDGE_TONGYI_PROMPT.format(
                question=question,
                response=solution_strs[i],
                correct_answer=ground_truths[i],
            )
            messages = [{"role": "user", "content": prompt}]
            for attempt in range(max_retries):
                ep = mgr.get_endpoint()
                client = _get_client(ep, api_key)
                try:
                    async with sem:
                        resp = await asyncio.wait_for(
                            client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                max_tokens=1024,
                                temperature=0.0,
                            ),
                            timeout=timeout,
                        )
                    mgr.report_success(ep)
                    output = (resp.choices[0].message.content or "").strip()
                    result = _parse_judge_output(output)
                    if result != "PARSE FAILED":
                        acc = result["accuracy"]
                        out = {"score": acc, "accuracy": acc, **result}
                        do_print = random.randint(1, 16) == 1
                        if do_print:
                            _log_judge_sample(
                                data_source=data_sources[i] if i < len(data_sources) else "",
                                extra_info=ex,
                                question=question,
                                solution_str=solution_strs[i],
                                ground_truth=ground_truths[i],
                                result=result,
                            )
                        return out
                    logger.warning("llm_judge batch parse failed sample %s attempt %s (endpoint %s)", i, attempt + 1, ep)
                except Exception as e:
                    mgr.report_failure(ep)
                    logger.warning("llm_judge batch sample %s attempt %s (endpoint %s): %s", i, attempt + 1, ep, e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(min((2 ** attempt) * random.uniform(0.8, 1.2), 10.0))
            return 0.0

        tasks = [_one(i) for i in range(len(solution_strs))]
        return list(await asyncio.gather(*tasks))

    return asyncio.run(_batch())


__all__ = [
    "compute_score",
    "compute_score_batch",
]
