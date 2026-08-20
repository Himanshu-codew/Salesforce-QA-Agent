"""
Qwen3 LLM implementation using OpenAI-compatible API.
Works with Groq (free), DashScope, Ollama, or any OpenAI-compatible provider.
Supports tool/function calling for the agent loop.
Strips <think>...</think> reasoning blocks from Qwen3 output.
"""

import json
import logging
import os
import re
from typing import Any

from openai import AsyncOpenAI

import httpx

from .base import BaseLLM

logger = logging.getLogger(__name__)


def strip_thinking(text: str) -> str:
    """
    Remove <think>...</think> blocks from Qwen3 model output.
    Qwen3 models produce internal reasoning in these tags — we strip
    them so only the clean response is shown to the user.
    """
    if not text:
        return text
    # Remove <think>...</think> blocks (including multiline and unclosed <think>...)
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL)
    return cleaned.strip()


def strip_raw_tool_json(text: str) -> str:
    """
    Remove raw JSON tool call blocks like { "name": "soqlQuery", "arguments": ... }
    from response text so raw JSON code never leaks into the user UI.
    """
    if not text:
        return text
    # 1. Remove markdown json blocks containing "name": "..."
    cleaned = re.sub(r"```(?:json)?\s*\{\s*\"name\"\s*:.*?\}(?:\s*```)?", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 2. Remove standalone { "name": "...", "arguments": { ... } } pattern
    cleaned = re.sub(r"\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:\s*\{.*?\}\s*\}", "", cleaned, flags=re.DOTALL)
    # 3. Remove XML tool tags <tools>...</tools> or <tool_call>...</tool_call>
    cleaned = re.sub(r"<(?:tools|tool_call)>.*?</(?:tools|tool_call)>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _get_headers(base_url: str) -> dict[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    if "ngrok" in base_url.lower():
        headers["bypass-tunnel-reminder"] = "true"
    elif "loca.lt" in base_url.lower():
        headers["bypass-tunnel-reminder"] = "true"
    return headers


def _extract_text_tool_calls(content: str) -> list[dict[str, Any]]:
    """Extract tool calls embedded as XML, markdown JSON blocks, or raw JSON in model output."""
    if not content:
        return []
    tool_calls = []

    def try_add_tool(data: Any) -> bool:
        if isinstance(data, list):
            added = False
            for item in data:
                if try_add_tool(item):
                    added = True
            return added

        if isinstance(data, dict):
            name = data.get("name") or data.get("function_name") or data.get("function")
            args = data.get("arguments") or data.get("parameters") or {}
            # Sometimes models return arguments as a stringified JSON
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            if name and isinstance(name, str):
                tool_calls.append({
                    "id": f"text_tc_{len(tool_calls)+1}",
                    "name": name.strip(),
                    "arguments": args if isinstance(args, dict) else {},
                })
                return True
        return False

    # 1. Try parsing the entire content as JSON directly
    try:
        data = json.loads(content.strip())
        if try_add_tool(data):
            return tool_calls
    except Exception:
        pass

    # 2. Look for markdown json blocks (```json ... ```)
    json_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    for block in json_blocks:
        try:
            data = json.loads(block.strip())
            try_add_tool(data)
        except Exception:
            pass

    if tool_calls:
        return tool_calls

    # 3. Existing XML fallback (<tools>...</tools>)
    matches = re.findall(r"<(?:tools|tool_call)>(.*?)</(?:tools|tool_call)>", content, re.DOTALL | re.IGNORECASE)
    for m in matches:
        try:
            data = json.loads(m.strip())
            try_add_tool(data)
        except Exception:
            pass

    if tool_calls:
        return tool_calls

    # 4. Smart bracket extraction (find outermost [ ... ] or { ... })
    for start_char, end_char in [('[', ']'), ('{', '}')]:
        start_idx = content.find(start_char)
        if start_idx != -1:
            end_idx = content.rfind(end_char)
            if end_idx > start_idx:
                try:
                    data = json.loads(content[start_idx:end_idx+1])
                    if try_add_tool(data):
                        return tool_calls
                except Exception:
                    pass

    return tool_calls


class QwenLLM(BaseLLM):
    """
    Qwen3 model accessed via any OpenAI-compatible API.
    Tested with: Groq (free), DashScope, Ollama, OpenRouter.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        model: str = "qwen/qwen3.6-27b",
    ):
        self.model = model
        self.base_url = base_url
        timeout_sec = float(os.getenv("LLM_TIMEOUT", "600.0"))
        http_client = httpx.AsyncClient(
            verify=False,
            timeout=httpx.Timeout(timeout_sec, connect=15.0),
        )
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=timeout_sec,
            http_client=http_client,
            default_headers=_get_headers(base_url),
        )
        logger.info(f"QwenLLM initialized with model={model}, base_url={base_url}, timeout={timeout_sec}s")

    def _check_and_update_base_url(self) -> None:
        """Dynamically reload .env and update base_url, model, or api_key if changed in .env."""
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
            new_url = os.getenv("QWEN_BASE_URL")
            new_model = os.getenv("QWEN_MODEL")
            new_key = os.getenv("QWEN_API_KEY", "ollama")

            recreate = False
            if new_url and new_url != self.base_url:
                logger.info(f"🔄 Auto-updating QWEN_BASE_URL: {self.base_url} -> {new_url}")
                self.base_url = new_url
                recreate = True

            if new_model and new_model != self.model:
                logger.info(f"🔄 Auto-updating QWEN_MODEL: {self.model} -> {new_model}")
                self.model = new_model
                recreate = True

            if recreate:
                timeout_sec = float(os.getenv("LLM_TIMEOUT", "600.0"))
                http_client = httpx.AsyncClient(
                    verify=False,
                    timeout=httpx.Timeout(timeout_sec, connect=15.0),
                )
                self._client = AsyncOpenAI(
                    api_key=new_key if new_key else "ollama",
                    base_url=self.base_url,
                    max_retries=0,
                    timeout=timeout_sec,
                    http_client=http_client,
                    default_headers=_get_headers(self.base_url),
                )
        except Exception as e:
            logger.warning(f"Failed to check base_url update: {e}")

    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Send a simple chat completion request (no tools)."""
        self._check_and_update_base_url()
        try:
            extra_body = {}
            if any(k in self.base_url for k in ["localhost", "ngrok", "trycloudflare", "127.0.0.1"]):
                extra_body["options"] = {"num_ctx": 8192, "num_predict": 1024, "temperature": 0.0, "num_gpu": 100}

            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=1024,
                extra_body=extra_body if extra_body else None,
            )
            content = response.choices[0].message.content or ""
            content = strip_thinking(content)
            logger.debug(f"Chat response (truncated): {content[:200]}")
            return content

        except Exception as e:
            logger.error(
                f"Qwen chat error: {e}. "
                f"Check if URL is reachable: {self.base_url}"
            )
            raise ConnectionError(
                f"Failed to connect to LLM at '{self.base_url}'. "
                "If using Kaggle/Ngrok, check if your tunnel URL has expired or update QWEN_BASE_URL in .env."
            ) from e

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """
        Send a chat completion with tool/function calling.
        Returns parsed response with content, tool_calls, and finish_reason.
        """
        self._check_and_update_base_url()
        try:
            extra_body = {}
            if any(k in self.base_url for k in ["localhost", "ngrok", "trycloudflare", "127.0.0.1"]):
                target_tokens = min(max_tokens, 1024) if tools else max_tokens
                extra_body["options"] = {"num_ctx": 8192, "num_predict": target_tokens, "temperature": 0.0, "num_gpu": 100}

            # Sanitize None content to empty string for Ollama/GGUF template compatibility
            clean_messages = []
            for m in messages:
                msg_copy = dict(m)
                if msg_copy.get("content") is None:
                    msg_copy["content"] = ""
                clean_messages.append(msg_copy)

            response = await self._client.chat.completions.create(
                model=self.model,
                messages=clean_messages,
                tools=tools if tools else None,
                temperature=0.0,
                max_tokens=min(max_tokens, 1024) if tools else max_tokens,
                extra_body=extra_body if extra_body else None,
            )

            choice = response.choices[0]
            message = choice.message
            result: dict[str, Any] = {
                "content": strip_thinking(message.content) if message.content else message.content,
                "tool_calls": [],
                "finish_reason": choice.finish_reason,
            }

            # Parse native tool calls if present
            if message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = tc.function.arguments

                    result["tool_calls"].append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": arguments,
                    })
            elif result["content"]:
                # Fallback: Extract embedded XML/JSON tool calls from Qwen3 GGUF text output
                extracted = _extract_text_tool_calls(result["content"])
                if extracted:
                    result["tool_calls"].extend(extracted)
                    result["content"] = strip_raw_tool_json(result["content"])

            # Clean any leftover raw JSON tool call syntax from text response
            if result["content"]:
                result["content"] = strip_raw_tool_json(result["content"])

            if result["tool_calls"]:
                logger.info(
                    f"Tool calls requested: "
                    f"{[tc['name'] for tc in result['tool_calls']]}"
                )

            return result

        except Exception as e:
            logger.error(
                f"Qwen chat_with_tools error: {e}. "
                f"Check if URL is reachable: {self.base_url}"
            )
            raise ConnectionError(
                f"Failed to connect to LLM at '{self.base_url}'. "
                "If using Kaggle/Ngrok, check if your tunnel URL has expired or update QWEN_BASE_URL in .env."
            ) from e

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        try:
            await self._client.close()
            logger.info("QwenLLM client closed.")
        except Exception as e:
            logger.warning(f"Error closing QwenLLM client: {e}")
