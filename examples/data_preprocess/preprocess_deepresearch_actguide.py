# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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
DeepResearch preprocessing for action-guidance (ActGuide) RL.

Produces an unguided student prompt while storing the full action trajectory in
``extra_info`` so that the teacher's guided prompt can be reconstructed at
training time from ``extra_info["all_actions"]`` and
``extra_info["base_question"]`` via
``verl/trainer/ppo/adaptive_guidance.rebuild_raw_prompt_with_k``.

Usage:
  python preprocess_deepresearch_actguide.py \\
    --input_jsonl path/to/iter.jsonl \\
    --output_path output/deepresearch_actguide.parquet
"""

import argparse
import json
import logging
import os
from typing import List, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_CONTENT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Current date: 2026-03-10"""

# Unguided prompt: no mention of action trajectory hints.
DEFAULT_QUESTION_CONTENT = """Answer the given question using the given tools.
For each step, you must conduct a thought section to reason before calling any tools.
Question: {question}"""


# ---------------------------------------------------------------------------
# Tool call extraction (from v5/v3)
# ---------------------------------------------------------------------------

ALLOWED_TOOL_NAMES = {"search", "visit"}


def extract_tool_calls_from_messages(messages: list) -> List[Tuple[str, dict]]:
    """Extract (name, arguments_dict) tuples from <tool_call> blocks in assistant messages.

    Only keeps actions whose name is in ALLOWED_TOOL_NAMES (search, visit).
    """
    result: List[Tuple[str, dict]] = []
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    for msg in messages or []:
        if (msg.get("role") or "").lower() != "assistant":
            continue
        content = msg.get("content") or ""
        pos = 0
        while True:
            i = content.find(start_tag, pos)
            if i == -1:
                break
            j = content.find(">", i)
            k = content.find(end_tag, j)
            if k == -1:
                break
            payload = content[j + 1 : k].strip()
            pos = k + len(end_tag)
            try:
                data = json.loads(payload)
                name = (data.get("name") or "").strip()
                args = data.get("arguments")
                if not isinstance(args, dict):
                    args = {}
                if name and name in ALLOWED_TOOL_NAMES:
                    result.append((name, args))
            except (json.JSONDecodeError, TypeError):
                continue
    return result


# ---------------------------------------------------------------------------
# Action serialization helpers (from v5)
# ---------------------------------------------------------------------------

def _format_tool_args_for_hint(name: str, args: dict) -> dict:
    """Strip 'goal' from visit actions for hint display."""
    if name == "visit":
        return {k: v for k, v in args.items() if k != "goal"}
    return args


def _action_to_dict(name: str, args: dict) -> dict:
    """Convert (name, args) tuple to a serializable dict for storage in extra_info."""
    cleaned = {k: v for k, v in _format_tool_args_for_hint(name, args).items() if v is not None}
    return {"name": name, "args": cleaned}


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def _get_question(obj: dict) -> str:
    question = (obj.get("question") or "").strip()
    if question:
        return question
    for msg in obj.get("messages") or []:
        if (msg.get("role") or "").lower() == "user":
            content = (msg.get("content") or "").strip()
            if "<tool_response>" not in content:
                return content
    return ""


def _get_answer(obj: dict) -> str:
    return (obj.get("answer") or "").strip()


# ---------------------------------------------------------------------------
# Row processing
# ---------------------------------------------------------------------------

def process_single_row(
    obj: dict,
    idx: int,
    *,
    split: str,
    data_source: str,
    system_content: str,
    question_content_template: str,
    search_tool_name: str,
    visit_tool_name: str,
) -> dict:
    question = _get_question(obj)
    answer = _get_answer(obj)

    # Extract action trajectory from messages for teacher distillation
    messages = obj.get("messages") or []
    extracted_actions = extract_tool_calls_from_messages(messages)
    n_actions = len(extracted_actions)

    all_actions_list = [_action_to_dict(name, args) for name, args in extracted_actions]
    all_actions = json.dumps(all_actions_list, ensure_ascii=False)

    # Student prompt: unguided (no action hints)
    user_content = question_content_template.format(question=question)
    prompt = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    reward_model = {
        "style": "rule",
        "ground_truth": answer,
    }

    _empty_create_kwargs = {"_": None}
    tools_kwargs = {
        search_tool_name: {"create_kwargs": _empty_create_kwargs},
        visit_tool_name: {"create_kwargs": _empty_create_kwargs},
    }

    extra_info = {
        "split": split,
        "index": idx,
        "question": question,
        "answer": answer,
        "need_tools_kwargs": True,
        "tools_kwargs": tools_kwargs,
        "all_actions": all_actions,
        "base_question": question,
        "n_total_actions": n_actions,
    }

    return {
        "data_source": data_source,
        "agent_name": "tool_agent",
        "prompt": prompt,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    logger.info("Reading JSONL: %s", args.input_jsonl)

    raw_data = []
    dropped = 0
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid line %d: %s", line_no + 1, e)
                dropped += 1
                continue
            if obj.get("error"):
                dropped += 1
                continue
            if not _get_question(obj):
                logger.warning("Skipping line %d with no question", line_no + 1)
                dropped += 1
                continue
            raw_data.append(obj)

    total = len(raw_data)
    logger.info("Valid samples: %d, dropped: %d", total, dropped)

    if total == 0:
        logger.error("No valid data, exiting")
        return

    data = []
    for idx, obj in enumerate(raw_data):
        row = process_single_row(
            obj,
            idx,
            split=args.split,
            data_source=args.data_source,
            system_content=args.system_content,
            question_content_template=args.question_content,
            search_tool_name=args.search_tool_name,
            visit_tool_name=args.visit_tool_name,
        )
        data.append(row)

    n_values = [d["extra_info"]["n_total_actions"] for d in data]
    logger.info(
        "On-policy distillation v6: samples=%d, n_actions in [%d, %d], mean=%.1f",
        total,
        min(n_values),
        max(n_values),
        sum(n_values) / len(n_values),
    )

    df = pd.DataFrame(data)
    out_path = args.output_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "DeepResearch preprocessing v6: unguided student prompt with "
            "action trajectory stored for on-policy distillation."
        )
    )
    parser.add_argument("--input_jsonl", required=True, help="Input JSONL file path")
    parser.add_argument("--output_path", required=True, help="Output parquet path")
    parser.add_argument("--split", default="test", help="Dataset split name")
    parser.add_argument("--data_source", default="DeepResearch", help="Data source label")
    parser.add_argument(
        "--system_content",
        default=DEFAULT_SYSTEM_CONTENT,
        help="System prompt content",
    )
    parser.add_argument(
        "--question_content",
        default=DEFAULT_QUESTION_CONTENT,
        help="User message template with {question} placeholder",
    )
    parser.add_argument("--search_tool_name", default="search", help="Search tool name")
    parser.add_argument("--visit_tool_name", default="visit", help="Visit tool name")
    args = parser.parse_args()
    main(args)
