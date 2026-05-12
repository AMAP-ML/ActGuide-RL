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

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import ray
import requests

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from .map_tool import PoolMode, init_search_execution_pool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Extra seconds on top of request timeout for Ray get (avoids training stuck when API/worker hangs)
RAY_GET_TIMEOUT_EXTRA = 15


async def _get_remote_with_timeout(ref, timeout_sec: int) -> Tuple[Any, Any]:
    """Wait for Ray remote result with a hard timeout so training never blocks.
    Returns (result_text, metadata). On timeout or failure returns (error_json_str, error_metadata).
    """
    loop = asyncio.get_running_loop()
    total_timeout = timeout_sec + RAY_GET_TIMEOUT_EXTRA

    def _get():
        return ray.get(ref, timeout=timeout_sec)

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _get),
            timeout=total_timeout,
        )
        return result
    except asyncio.TimeoutError:
        err_msg = "Tool call timed out (worker or API did not respond in time)."
        logger.warning("[deepresearch_tool] %s", err_msg)
        return (
            json.dumps({"result": err_msg}),
            {"status": "timeout", "error": err_msg},
        )
    except Exception as e:
        logger.warning("[deepresearch_tool] Ray get failed: %s", e)
        return (
            json.dumps({"result": f"Tool call failed: {e}"}),
            {"status": "error", "error": str(e)},
        )


class SearchTool(BaseTool):
    """Search tool: calls a locally deployed service (e.g. 127.0.0.1:xxxx) for web search.

    Config must include 'url' pointing to the local search endpoint (e.g. http://127.0.0.1:8001/search).
    The service is expected to accept POST with {"query": "..."} and return the search result (e.g. JSON with results).
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: Dict[str, Dict[str, Any]] = {}
        self.url = config.get("url")
        assert self.url, "Configuration must include 'url' (e.g. http://127.0.0.1:8001/search)"
        if self.url == "":
            raise ValueError("url is not set")
        self.timeout = config.get("timeout", 30)
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        logger.info("Initialized SearchTool with url=%s", self.url)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"response": "", "reward": []}
        return instance_id, ToolResponse()

    def execute_api_call(self, instance_id: str, query: str, url: str, timeout: int):
        """Call local search service: POST with {"query": query}. Returns (result_text, metadata)."""
        payload = {"query": query}
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            result_text = response.text
            metadata = {"status": "success", "total_results": 0}
            return result_text, metadata
        except requests.exceptions.RequestException as e:
            logger.warning("Search API call failed: %s", e)
            result_text = json.dumps({"result": f"Search API call failed: {e}"})
            return result_text, {"status": "api_error", "error": str(e)}

    def _normalize_queries(self, query: Any) -> List[str]:
        """Normalize to list of non-empty strings. Accept single string or list of strings."""
        if isinstance(query, str) and query.strip():
            return [query.strip()]
        if isinstance(query, list):
            return [str(q).strip() for q in query if q is not None and str(q).strip()]
        return []

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        raw = parameters.get("query")
        queries = self._normalize_queries(raw)
        if not queries:
            err = "Error: 'query' is missing, empty, or not a string/list of strings in parameters."
            logger.error("[search] %s Received: %s", err, parameters)
            return ToolResponse(text=json.dumps({"result": err})), 0.0, {}

        results_list: List[Any] = []
        all_metadata: Dict[str, Any] = {"status": "success", "total_results": 0}
        try:
            for q in queries:
                ref = self.execution_pool.execute.remote(
                    self.execute_api_call, instance_id, q, self.url, self.timeout
                )
                result_text, metadata = await _get_remote_with_timeout(ref, self.timeout)
                results_list.append({"query": q, "result": result_text})
                if isinstance(result_text, str):
                    self._instance_dict[instance_id]["reward"].append(result_text.strip())
                all_metadata["total_results"] = all_metadata.get("total_results", 0) + 1
            combined = json.dumps({"results": results_list}) if results_list else "{}"
            return ToolResponse(text=combined), 0.0, all_metadata
        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error("[search] Execution failed: %s", e)
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> Any:
        return self._instance_dict.get(instance_id, {}).get("reward", [])

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)


class VisitTool(BaseTool):
    """Visit (crawl) tool: calls a locally deployed service (e.g. 127.0.0.1:xxxx) to fetch page content.

    Config must include 'url' pointing to the local visit/crawl endpoint (e.g. http://127.0.0.1:8001/visit).
    The service is expected to accept POST with {"url": "https://...", "goal": "..."} and return the summary of the content.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: Dict[str, Dict[str, Any]] = {}
        self.url = config.get("url")
        assert self.url, "Configuration must include 'url' (e.g. http://127.0.0.1:8001/visit)"
        if self.url == "":
            raise ValueError("url is not set")
        self.timeout = config.get("timeout", 50)
        # 同一 (url, goal) 最多发起几次远端请求；超出则本 episode 内直接返回提示，不再 HTTP
        self.max_repeat_per_url_goal = int(config.get("max_repeat_per_url_goal", 2))
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        logger.info(
            "Initialized VisitTool with url=%s max_repeat_per_url_goal=%s",
            self.url,
            self.max_repeat_per_url_goal,
        )

    def _filter_url(self, url: str) -> str:
        for prefix in ("https://r.jina.ai/", "view-source:"):
            if url.startswith(prefix):
                url = url[len(prefix) :]
                logger.debug("Filtered URL prefix -> %s", url)
                break
        return url

    def _normalize_urls(self, url: Any) -> List[str]:
        """Normalize to list of non-empty URL strings. Accept single string or list of strings."""
        if isinstance(url, str) and url.strip():
            return [self._filter_url(url.strip())]
        if isinstance(url, list):
            return [self._filter_url(str(u).strip()) for u in url if u is not None and str(u).strip()]
        return []

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
            "visit_url_goal_attempts": {},
        }
        return instance_id, ToolResponse()

    def execute_api_call(
        self,
        instance_id: str,
        page_url: str,
        goal: str,
        service_url: str,
        timeout: int,
    ):
        """Call local visit service: POST with {"url": page_url, "goal": goal}. Returns (result_text, metadata)."""
        payload = {"url": page_url, "goal": goal}
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(
                service_url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()
            result_text = response.text
            metadata = {"status": "success"}
            return result_text, metadata
        except requests.exceptions.RequestException as e:
            logger.warning("Visit API call failed: %s", e)
            result_text = json.dumps(
                {"result": "[visit] Failed to read page.", "url": page_url}
            )
            return result_text, {"status": "api_error", "error": str(e)}

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        raw = parameters.get("url")
        urls = self._normalize_urls(raw)
        if not urls:
            err = "Error: 'url' is missing, empty, or not a string/list of strings in parameters."
            logger.error("[visit] %s Received: %s", err, parameters)
            return ToolResponse(text=json.dumps({"result": err})), 0.0, {}

        goal = parameters.get("goal")
        if goal is None or (isinstance(goal, str) and not goal.strip()):
            err = "Error: 'goal' is required and must be a non-empty string (the specific information goal for visiting webpage(s))."
            logger.error("[visit] %s Received: %s", err, parameters)
            return ToolResponse(text=json.dumps({"result": err})), 0.0, {}

        goal_str = goal.strip() if isinstance(goal, str) else str(goal).strip()

        inst = self._instance_dict.setdefault(
            instance_id,
            {"response": "", "reward": [], "visit_url_goal_attempts": {}},
        )
        attempts_map: Dict[str, int] = inst.setdefault("visit_url_goal_attempts", {})

        results_list: List[Any] = []
        all_metadata: Dict[str, Any] = {"status": "success"}
        try:
            for u in urls:
                ug_key = f"{u}\x00{goal_str}"
                if attempts_map.get(ug_key, 0) >= self.max_repeat_per_url_goal:
                    msg = json.dumps(
                        {
                            "result": (
                                f"[visit] Same URL+goal already attempted "
                                f"{self.max_repeat_per_url_goal} time(s) in this trajectory; skipping."
                            ),
                            "url": u,
                        }
                    )
                    results_list.append({"url": u, "result": msg})
                    continue
                attempts_map[ug_key] = attempts_map.get(ug_key, 0) + 1

                ref = self.execution_pool.execute.remote(
                    self.execute_api_call,
                    instance_id,
                    u,
                    goal_str,
                    self.url,
                    self.timeout,
                )
                result_text, metadata = await _get_remote_with_timeout(ref, self.timeout)
                results_list.append({"url": u, "result": result_text})
                if isinstance(result_text, str):
                    self._instance_dict[instance_id]["reward"].append(result_text[:500])
            combined = json.dumps({"results": results_list}) if results_list else "{}"
            return ToolResponse(text=combined), 0.0, all_metadata
        except Exception as e:
            error_result = json.dumps(
                {"result": "[visit] Failed to read page.", "urls": urls}
            )
            logger.error("[visit] Execution failed: %s", e)
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> Any:
        return self._instance_dict.get(instance_id, {}).get("reward", [])

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
