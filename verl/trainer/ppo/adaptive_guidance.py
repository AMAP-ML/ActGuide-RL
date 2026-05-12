# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Adaptive action guidance via training-time binary search.

Thin orchestration layer (~120 lines) that reuses existing infrastructure:
  - DataProto.select_idxs  for efficient sub-batch selection
  - generate_sequences     for full agent loop (tool_agent_loop + inference engine)
  - compute_reward         for reward evaluation

Design follows the REMAX baseline pattern in ray_trainer.py lines 1069-1092.
"""

import copy
import json
import logging
from typing import Any, Callable

import numpy as np

from verl import DataProto
from verl.trainer.ppo.reward import compute_reward

logger = logging.getLogger(__name__)

# Same template used in examples/data_preprocess/preprocess_deepresearch_actguide.py
QUESTION_TEMPLATE = """Answer the given question using the given tools.
For each step, you must conduct a thought section to reason before calling any tools.
Question: {question}
Follow the partial action trajectory hint to take acttions, note that the trajectory may not complete and you still need do some extra tool calls to finish the task.
Reference action trajectory hint:
{action_trajectory_hint}"""

APPEND_ONLY_HINT_TEMPLATE = """{question}
Follow the partial action trajectory hint to take acttions, note that the trajectory may not complete and you still need do some extra tool calls to finish the task.
Reference action trajectory hint:
{action_trajectory_hint}"""

ASSISTANT_PREFIX_TEMPLATE = """I have a partial action plan trajectory hint:
{action_trajectory_hint}

I will solve the task by following this hint when useful.
"""

# assistant_prefix 模式下使用的 user content 模板，不含 hint 部分，
# 避免 user content 中的 "hint: None" 与 assistant prefix 中的实际 hint 矛盾。
CLEAN_QUESTION_TEMPLATE = """Answer the given question using the given tools.
For each step, you must conduct a thought section to reason before calling any tools.
Question: {question}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_all_actions(raw) -> list[dict]:
    """Parse all_actions from extra_info.

    Handles three storage formats:
      - JSON string (v5 parquet): '[ {"name":"search","args":{...}}, ... ]'
      - list[dict] (in-memory / test):  [{"name":"search","args":{...}}, ...]
      - numpy array of dicts (old v5 parquet before JSON-string fix)
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return []
    if isinstance(raw, np.ndarray):
        return [dict(item) if hasattr(item, "items") else item for item in raw]
    if isinstance(raw, list):
        return raw
    return []


# ---------------------------------------------------------------------------
# Prompt reconstruction utilities
# ---------------------------------------------------------------------------

def build_hint_from_actions(all_actions: list[dict], k: int) -> str:
    """Build numbered hint string from the first *k* actions."""
    if k <= 0 or not all_actions:
        return "None"
    lines = [
        f"{i}. {a['name']}: {json.dumps(a['args'], ensure_ascii=False)}"
        for i, a in enumerate(all_actions[:k], 1)
    ]
    return "\n".join(lines)


def _append_hint_only_suffix(hint: str) -> str:
    return (
        "Follow the partial action trajectory hint to take acttions, note that the trajectory may not complete "
        "and you still need do some extra tool calls to finish the task.\n"
        "Reference action trajectory hint:\n"
        f"{hint}"
    )


def _append_hint_to_multimodal_content(content, hint: str):
    """Preserve multimodal content list and append hint to the last text chunk."""
    suffix = _append_hint_only_suffix(hint)
    if not isinstance(content, list):
        return APPEND_ONLY_HINT_TEMPLATE.format(question=content, action_trajectory_hint=hint)

    new_content = copy.deepcopy(content)
    for idx in range(len(new_content) - 1, -1, -1):
        item = new_content[idx]
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text", "")
            sep = "\n" if text and not text.endswith("\n") else ""
            item["text"] = f"{text}{sep}{suffix}"
            return new_content

    new_content.append({"type": "text", "text": suffix})
    return new_content


def rebuild_raw_prompt_with_k(
    raw_prompt: list[dict],
    base_question: str,
    all_actions: list[dict],
    k: int,
    injection_mode: str = "user_prompt",
    prompt_style: str = "default",
) -> list[dict]:
    """Return a NEW messages list with guidance injected in the configured mode."""
    new_prompt = copy.deepcopy(raw_prompt)
    hint = build_hint_from_actions(all_actions, k)
    if injection_mode == "assistant_prefix":
        # 移除之前可能已注入的 assistant prefix（避免多次调用时重复追加）
        while len(new_prompt) > 1 and new_prompt[-1]["role"] == "assistant":
            new_prompt.pop()
        # 清理 user content 中可能残留的 hint 模板（v5 数据含 "hint: None"），
        # 避免与 assistant prefix 中的实际 hint 矛盾
        if new_prompt[-1]["role"] == "user":
            if prompt_style == "append_hint_only":
                pass
            else:
                new_prompt[-1]["content"] = CLEAN_QUESTION_TEMPLATE.format(question=base_question)
        if hint == "None":
            return new_prompt
        new_prompt.append(
            {
                "role": "assistant",
                "content": ASSISTANT_PREFIX_TEMPLATE.format(action_trajectory_hint=hint),
            }
        )
        return new_prompt
    if prompt_style == "append_hint_only":
        new_prompt[-1]["content"] = _append_hint_to_multimodal_content(
            new_prompt[-1]["content"],
            hint,
        )
    else:
        new_prompt[-1]["content"] = QUESTION_TEMPLATE.format(
            question=base_question,
            action_trajectory_hint=hint,
        )
    return new_prompt


# ---------------------------------------------------------------------------
# AdaptiveGuidanceSearcher
# ---------------------------------------------------------------------------

class AdaptiveGuidanceSearcher:
    """Binary-search for the optimal action guidance length before each rollout.

    Uses the monotonicity assumption: P(reward > 0) is non-decreasing in k
    (the number of action hints provided).
    """

    def __init__(
        self,
        enabled: bool = True,
        max_probe_steps: int = 4,
        k_offset: int = 0,
        probe_rollouts: int = 1,
        injection_mode: str = "user_prompt",
        **kwargs,
    ):
        self.max_probe_steps = max_probe_steps
        self.k_offset = k_offset
        self.probe_rollouts = max(1, int(probe_rollouts))
        self.injection_mode = injection_mode

    def search(
        self,
        gen_batch: DataProto,
        batch: DataProto,
        generate_fn: Callable,
        reward_fn: Any,
        reward_success_threshold: float = 0.0,
    ) -> list[int]:
        """Binary-search optimal *k* for every sample in the batch.

        Returns a list of optimal k values (one per sample).
        Samples without ``all_actions`` in extra_info are assigned k=0.
        """
        N = len(gen_batch)
        extra_infos = gen_batch.non_tensor_batch["extra_info"]
        all_actions = [_parse_all_actions(info.get("all_actions", [])) for info in extra_infos]

        lo = [0] * N
        hi = [len(a) for a in all_actions]

        for step in range(self.max_probe_steps):
            active = [i for i in range(N) if lo[i] < hi[i] and len(all_actions[i]) > 0]
            if not active:
                break

            mid_map = {i: (lo[i] + hi[i]) // 2 for i in active}

            # Probe with the same sampling behavior as training rollout.
            # We intentionally do not force do_sample=False here.
            reward_success = np.zeros(N, dtype=bool)
            for probe_idx in range(self.probe_rollouts):
                # Use full batch (same as REMAX baseline pattern) to avoid
                # chunk/padding issues with agent loop workers.
                probe = copy.deepcopy(gen_batch)

                # Modify raw_prompt only for active (unconverged) samples
                for idx in active:
                    probe.non_tensor_batch["raw_prompt"][idx] = rebuild_raw_prompt_with_k(
                        probe.non_tensor_batch["raw_prompt"][idx],
                        extra_infos[idx]["base_question"],
                        all_actions[idx],
                        mid_map[idx],
                        injection_mode=self.injection_mode,
                        prompt_style=extra_infos[idx].get("guidance_prompt_style", "default"),
                    )

                # Probe rollout -- fully reuses tool_agent_loop + inference engine
                probe_output = generate_fn(probe)

                # Attach reward metadata so compute_reward can evaluate
                for key in ("reward_model", "data_source", "extra_info"):
                    if key in probe.non_tensor_batch:
                        probe_output.non_tensor_batch[key] = probe.non_tensor_batch[key]

                reward_tensor, _ = compute_reward(probe_output, reward_fn)
                rewards = reward_tensor.sum(dim=-1).detach().cpu().numpy()
                reward_success |= rewards > reward_success_threshold
                del probe, probe_output

            # Update binary-search bounds (only for active samples).
            # Explicit any-success rule: if any probe rollout exceeds the
            # success threshold, we treat this k as successful.
            for idx in active:
                if reward_success[idx]:
                    hi[idx] = mid_map[idx]
                else:
                    lo[idx] = mid_map[idx] + 1

            n_still_active = sum(1 for i in active if lo[i] < hi[i])
            logger.info(
                "Adaptive guidance probe step %d/%d: active=%d -> %d, probe_rollouts=%d, success_threshold=%.4f",
                step + 1,
                self.max_probe_steps,
                len(active),
                n_still_active,
                self.probe_rollouts,
                reward_success_threshold,
            )

        optimal_k = [
            max(0, min(lo[i] + self.k_offset, len(all_actions[i])))
            for i in range(N)
        ]
        return optimal_k

    def apply(
        self,
        gen_batch: DataProto,
        batch: DataProto,
        optimal_k: list[int],
    ) -> None:
        """Modify *gen_batch* ``raw_prompt`` in-place with the chosen guidance level."""
        extra_infos = gen_batch.non_tensor_batch["extra_info"]
        for i in range(len(gen_batch)):
            actions = _parse_all_actions(extra_infos[i].get("all_actions"))
            if not actions:
                continue
            gen_batch.non_tensor_batch["raw_prompt"][i] = rebuild_raw_prompt_with_k(
                gen_batch.non_tensor_batch["raw_prompt"][i],
                extra_infos[i]["base_question"],
                actions,
                optimal_k[i],
                injection_mode=self.injection_mode,
                prompt_style=extra_infos[i].get("guidance_prompt_style", "default"),
            )
