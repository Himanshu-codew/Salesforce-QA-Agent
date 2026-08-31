"""
Qwen LLM implementation using OpenAI-compatible API.
Works with Groq (free), DashScope, Ollama, or any OpenAI-compatible provider.
Supports tool/function calling for the agent loop.
Strips <think>...</think> reasoning blocks from Qwen3 output.
"""

import json
import logging
import os
import re
from typing import Any

from openai import AsyncOpenAI, APIStatusError, NotFoundError, APIConnectionError, RateLimitError

import httpx

from .base import BaseLLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions for clear error classification
# ---------------------------------------------------------------------------

class LLMConnectionError(Exception):
    """Network-level failure: host unreachable, DNS error, tunnel down, timeout."""


class LLMModelError(Exception):
    """Model-level failure: 404 model not found, 400 bad request from API, etc."""


class LLMToolError(Exception):
    """Tool-calling compatibility error: native tools unsupported and fallback also failed."""


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """
    Remove <think>...</think> blocks from Qwen3 model output.
    Qwen3 models produce internal reasoning in these tags — we strip
    them so only the clean response is shown to the user.
    """
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL)
    return cleaned.strip()


def strip_raw_tool_json(text: str) -> str:
    """
    Remove raw JSON tool call blocks like { "name": "soqlQuery", "arguments": ... }
    from response text so raw JSON code or stray braces never leak into the user UI.
    This is the LLM-level sanitizer; agent.py applies a second pass before delivery.
    """
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?[\s\S]*?\"name\"[\s\S]*?```", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<(?:tools|tool_call|function_call)>[\s\S]*?</(?:tools|tool_call|function_call)>", "", cleaned, flags=re.IGNORECASE)

    known_tools = (
        "soqlQuery|find|getUserInfo|getObjectSchema|createSobjectRecord|"
        "updateSobjectRecord|deleteSobjectRecord|getRelatedRecords|"
        "listRecentSobjectRecords|updateRelatedRecord|deleteRelatedRecord|"
        "uploadRecordAttachment"
    )
    cleaned = re.sub(
        rf'\{{\s*"name"\s*:\s*"(?:{known_tools})"[\s\S]*?\}}\s*\}}\s*\}}',
        "", cleaned
    )
    cleaned = re.sub(
        rf'\{{\s*"name"\s*:\s*"(?:{known_tools})"[\s\S]*?\}}\s*\}}',
        "", cleaned
    )
    cleaned = re.sub(
        rf'\{{\s*"name"\s*:\s*"(?:{known_tools})"[\s\S]*?\}}',
        "", cleaned
    )
    cleaned = re.sub(
        rf'\{{\s*"function"\s*:\s*\{{\s*"name"\s*:\s*"(?:{known_tools})"[\s\S]*?\}}\s*\}}',
        "", cleaned
    )

    lines = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        if stripped in ["{", "}", "]", "[", "```", "```json", "}}", "}}}", "`]", "{}", "[]"]:
            continue
        if re.match(r'^\s*"(name|type|function|arguments|parameters|properties|required)"\s*:', stripped):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)

    final_text = cleaned.strip()
    if re.fullmatch(r"[\{\}\[\]\`\s]*", final_text):
        return ""
    return final_text


def _get_headers(base_url: str) -> dict[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    if "ngrok" in base_url.lower():
        headers["ngrok-skip-browser-warning"] = "true"
        headers["bypass-tunnel-reminder"] = "true"
    elif "loca.lt" in base_url.lower():
        headers["bypass-tunnel-reminder"] = "true"
    return headers


def _clean_json_str(s: str) -> str:
    """Strip C-style comments (// and /* */) and trailing commas before json.loads()."""
    if not s:
        return s
    s = re.sub(r"//.*", "", s)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r",\s*([\}\]])", r"\1", s)
    return s.strip()


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
            if isinstance(args, str):
                try:
                    args = json.loads(_clean_json_str(args))
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

    try:
        data = json.loads(_clean_json_str(content.strip()))
        if try_add_tool(data):
            return tool_calls
    except Exception:
        pass

    json_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    for block in json_blocks:
        try:
            data = json.loads(_clean_json_str(block.strip()))
            try_add_tool(data)
        except Exception:
            pass

    if tool_calls:
        return tool_calls

    matches = re.findall(r"<(?:tools|tool_call)>(.*?)</(?:tools|tool_call)>", content, re.DOTALL | re.IGNORECASE)
    for m in matches:
        try:
            data = json.loads(_clean_json_str(m.strip()))
            try_add_tool(data)
        except Exception:
            pass

    if tool_calls:
        return tool_calls

    for start_char, end_char in [('[', ']'), ('{', '}')]:
        start_idx = content.find(start_char)
        if start_idx != -1:
            end_idx = content.rfind(end_char)
            if end_idx > start_idx:
                try:
                    data = json.loads(_clean_json_str(content[start_idx:end_idx+1]))
                    if try_add_tool(data):
                        return tool_calls
                except Exception:
                    pass

    known_tools = {
        "getUserInfo", "soqlQuery", "find", "getObjectSchema",
        "describeSObject", "listRecentRecords", "createSobjectRecord",
        "updateSobjectRecord", "deleteSobjectRecord", "getGlobalDescribe",
        "executeApex", "batchCreateRecords", "getRelatedRecords",
        "updateRelatedRecord", "deleteRelatedRecord", "uploadRecordAttachment"
    }

    md_list_matches = re.finditer(r"[-*]\s*`?([a-zA-Z0-9_]+)`?\s*(?:->|:|=>)\s*`?(.*?)`?(?:\n|$)", content)
    for m in md_list_matches:
        fn_name = m.group(1).strip()
        fn_arg = m.group(2).strip()
        if fn_name in known_tools:
            arg_key = "q"
            if fn_name == "getObjectSchema":
                arg_key = "sobject-name"
            elif "Record" in fn_name or "Attachment" in fn_name:
                arg_key = "id"

            tool_calls.append({
                "id": f"text_tc_{len(tool_calls)+1}",
                "name": fn_name,
                "arguments": {arg_key: fn_arg},
            })

    if tool_calls:
        return tool_calls

    cleaned_content = content.strip().strip("`'\" \n\r\t")
    bare_name = cleaned_content.rstrip("()").strip()
    if bare_name in known_tools:
        tool_calls.append({
            "id": f"text_tc_{len(tool_calls)+1}",
            "name": bare_name,
            "arguments": {},
        })
        return tool_calls

    fn_match = re.match(r"^(\w+)\s*\((.*)\)$", cleaned_content, re.DOTALL)
    if fn_match:
        fn_name = fn_match.group(1).strip()
        fn_args_raw = fn_match.group(2).strip()
        if fn_name in known_tools:
            parsed_args = {}
            if fn_args_raw:
                try:
                    parsed_args = json.loads("{" + fn_args_raw + "}")
                except Exception:
                    for kv_match in re.finditer(r'(\w+)\s*=\s*(["\'])(.*?)\2', fn_args_raw, re.DOTALL):
                        parsed_args[kv_match.group(1)] = kv_match.group(3)
            tool_calls.append({
                "id": f"text_tc_{len(tool_calls)+1}",
                "name": fn_name,
                "arguments": parsed_args,
            })
            return tool_calls

    return tool_calls


# ---------------------------------------------------------------------------
# Helpers for classifying OpenAI / httpx exceptions
# ---------------------------------------------------------------------------

def _is_model_not_found(exc: Exception) -> bool:
    """Return True when the error means the model is not available on the server."""
    if isinstance(exc, NotFoundError):
        return True
    name = type(exc).__name__
    if "NotFoundError" in name:
        return True
    msg = str(exc).lower()
    return ("404" in msg and ("model" in msg or "not found" in msg)) or "model_not_found" in msg


def _is_connection_error(exc: Exception) -> bool:
    """Return True for network-level / transport errors."""
    if isinstance(exc, (APIConnectionError, httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout)):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in ["connect", "timeout", "timed out", "dns", "unreachable", "tunnel", "refused"])


def _is_tools_unsupported(exc: Exception) -> bool:
    """Return True when the server explicitly rejects native tool calling."""
    msg = str(exc).lower()
    return "does not support tools" in msg or "tools" in msg and "not support" in msg


def _is_rate_limit(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    return "rate limit" in str(exc).lower() or "429" in str(exc)


# ---------------------------------------------------------------------------
# QwenLLM
# ---------------------------------------------------------------------------

class QwenLLM(BaseLLM):
    """
    Qwen model accessed via any OpenAI-compatible API.
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
                logger.info(f"Auto-updating QWEN_BASE_URL: {self.base_url} -> {new_url}")
                self.base_url = new_url
                recreate = True

            if new_model and new_model != self.model:
                logger.info(f"Auto-updating QWEN_MODEL: {self.model} -> {new_model}")
                self.model = new_model
                recreate = True

            if new_key and new_key != self._client.api_key:
                logger.info("Auto-updating QWEN_API_KEY")
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

    async def _auto_fallback_model(self) -> bool:
        """Query /models from server and auto-select the loaded model if current model returns 404."""
        try:
            models_resp = await self._client.models.list()
            available = [m.id for m in models_resp.data if m.id]
            if available and self.model not in available:
                logger.warning(
                    f"Model '{self.model}' not found in server models {available}. "
                    f"Auto-switching to '{available[0]}'"
                )
                self.model = available[0]
                return True
        except Exception as e:
            logger.warning(f"Could not auto-discover models: {e}")
        return False

    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> str:
        """Send a simple chat completion request (no tools)."""
        self._check_and_update_base_url()
        extra_body: dict = {}
        if "qwen3" in self.model.lower() or "qwen/qwen3" in self.model.lower():
            extra_body["enable_thinking"] = False

        for attempt in range(2):
            try:
                logger.debug(f"Qwen chat request started (attempt {attempt + 1})")
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body if extra_body else None,
                )

                if not response.choices:
                    raise LLMModelError(f"Empty choices in API response (HTTP 200 but no choices returned)")

                content = response.choices[0].message.content or ""
                content = strip_thinking(content)
                logger.debug(f"Qwen chat response received ({len(content)} chars)")
                return content

            except Exception as e:
                # Only retry on model-not-found (404)
                if attempt == 0 and _is_model_not_found(e):
                    switched = await self._auto_fallback_model()
                    if switched:
                        continue

                # Classify the error — do NOT hide real exceptions as connection errors
                if isinstance(e, (LLMModelError,)):
                    raise

                if _is_connection_error(e):
                    logger.error(
                        f"Qwen chat connection error: {type(e).__name__} - {e}. "
                        f"Check if URL is reachable: {self.base_url}"
                    )
                    raise LLMConnectionError(
                        f"Failed to connect to LLM at '{self.base_url}' "
                        f"({type(e).__name__}: {e}). "
                        "If using Kaggle/Ngrok, check if your tunnel URL has expired "
                        "or update QWEN_BASE_URL in .env."
                    ) from e

                if isinstance(e, APIStatusError):
                    logger.error(
                        f"Qwen chat API error (HTTP {e.status_code}): {type(e).__name__} - {e}"
                    )
                    raise LLMModelError(
                        f"LLM API error at '{self.base_url}' (HTTP {e.status_code}): {e}"
                    ) from e

                # Programming / parsing / unexpected errors — do NOT wrap as connection error
                logger.error(
                    f"Qwen chat unexpected error: {type(e).__name__} - {e}",
                    exc_info=True,
                )
                raise

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        """
        Send a chat completion with tool/function calling.
        Returns parsed response with content, tool_calls, and finish_reason.
        """
        self._check_and_update_base_url()
        extra_body: dict = {}
        if "qwen3" in self.model.lower() or "qwen/qwen3" in self.model.lower():
            extra_body["enable_thinking"] = False

        # Sanitize None content to empty string for template compatibility
        clean_messages = []
        for m in messages:
            msg_copy = dict(m)
            if msg_copy.get("content") is None:
                msg_copy["content"] = ""
            clean_messages.append(msg_copy)

        for attempt in range(2):
            response = None
            used_fallback = False

            try:
                # --- Step 1: Try native tool calling ---
                logger.debug(f"Qwen request started (attempt {attempt + 1})")
                try:
                    response = await self._client.chat.completions.create(
                        model=self.model,
                        messages=clean_messages,
                        tools=tools if tools else None,
                        temperature=0.0,
                        max_tokens=max_tokens,
                        extra_body=extra_body if extra_body else None,
                    )
                    logger.debug("Qwen response received (native tools)")
                except Exception as native_err:
                    # Only fall back when the server explicitly rejects tool calling
                    if _is_tools_unsupported(native_err) and tools:
                        logger.warning(
                            f"Native tool calling unsupported on {self.model}; "
                            f"using prompt-based fallback"
                        )
                        tools_schema_str = json.dumps(tools, indent=2)
                        fallback_msg = (
                            "\n\nYou have access to the following JSON tools. "
                            "To use a tool, output a raw JSON array like: "
                            '[{{"name": "toolName", "arguments": {{...}}}}].\n'
                            f"Tools:\n{tools_schema_str}"
                        )
                        fallback_messages = list(clean_messages)
                        if fallback_messages and fallback_messages[0]["role"] == "system":
                            fallback_messages[0]["content"] += fallback_msg
                        else:
                            fallback_messages.insert(
                                0, {"role": "system", "content": fallback_msg}
                            )
                        response = await self._client.chat.completions.create(
                            model=self.model,
                            messages=fallback_messages,
                            tools=None,
                            temperature=0.0,
                            max_tokens=max_tokens,
                            extra_body=extra_body if extra_body else None,
                        )
                        used_fallback = True
                        logger.debug("Qwen response received (prompt-based fallback)")
                    else:
                        # Not a tools-unsupported error — let the outer handler deal with it
                        raise

                # --- Step 2: Validate the response ---
                if response is None:
                    raise LLMModelError("No response object returned from API")

                if not response.choices:
                    raise LLMModelError(
                        f"Empty choices in API response (HTTP 200 but no choices returned)"
                    )

                choice = response.choices[0]
                message = choice.message

                if message is None:
                    raise LLMModelError("Message object is None in API response")

                result: dict[str, Any] = {
                    "content": strip_thinking(message.content) if message.content else "",
                    "tool_calls": [],
                    "finish_reason": choice.finish_reason,
                }

                # --- Step 3: Parse native tool calls if present ---
                if message.tool_calls:
                    logger.info(
                        f"Native tool calls detected: "
                        f"{[tc.function.name for tc in message.tool_calls]}"
                    )
                    for tc in message.tool_calls:
                        try:
                            arguments = json.loads(tc.function.arguments)
                        except (json.JSONDecodeError, TypeError):
                            arguments = tc.function.arguments
                            logger.warning(
                                f"Tool call '{tc.function.name}' arguments not valid JSON; "
                                f"passing raw string"
                            )

                        result["tool_calls"].append({
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": arguments,
                        })

                elif result["content"]:
                    # --- Step 4: Fallback — extract embedded tool calls from text ---
                    extracted = _extract_text_tool_calls(result["content"])
                    if extracted:
                        if used_fallback:
                            logger.info(
                                f"Fallback tool calls extracted: "
                                f"{[tc['name'] for tc in extracted]}"
                            )
                        else:
                            logger.info(
                                f"Text-embedded tool calls extracted: "
                                f"{[tc['name'] for tc in extracted]}"
                            )
                        result["tool_calls"].extend(extracted)
                        result["content"] = strip_raw_tool_json(result["content"])

                # Final content sanitization
                if result["content"]:
                    result["content"] = strip_raw_tool_json(result["content"])

                if result["tool_calls"]:
                    logger.info(
                        f"Tool calls requested: "
                        f"{[tc['name'] for tc in result['tool_calls']]} "
                        f"(finish_reason={result['finish_reason']})"
                    )
                else:
                    logger.debug(
                        f"No tool calls in response "
                        f"(content_len={len(result['content'])}, "
                        f"finish_reason={result['finish_reason']})"
                    )

                return result

            except Exception as e:
                # --- Outer error handler: classify and re-raise ---

                # Only retry on model-not-found (404)
                if attempt == 0 and _is_model_not_found(e):
                    switched = await self._auto_fallback_model()
                    if switched:
                        continue

                # Already a typed error from inner code — just re-raise
                if isinstance(e, (LLMModelError, LLMConnectionError, LLMToolError)):
                    raise

                # Network / transport errors
                if _is_connection_error(e):
                    logger.error(
                        f"Qwen chat_with_tools connection error: {type(e).__name__} - {e}. "
                        f"Check if URL is reachable: {self.base_url}"
                    )
                    raise LLMConnectionError(
                        f"Failed to connect to LLM at '{self.base_url}' "
                        f"({type(e).__name__}: {e}). "
                        "If using Kaggle/Ngrok, check if your tunnel URL has expired "
                        "or update QWEN_BASE_URL in .env."
                    ) from e

                # HTTP / API status errors (4xx, 5xx)
                if isinstance(e, APIStatusError):
                    logger.error(
                        f"Qwen chat_with_tools API error (HTTP {e.status_code}): "
                        f"{type(e).__name__} - {e}"
                    )
                    raise LLMModelError(
                        f"LLM API error at '{self.base_url}' (HTTP {e.status_code}): {e}"
                    ) from e

                # Programming / parsing / unexpected errors — NEVER wrap as connection error
                logger.error(
                    f"Qwen chat_with_tools unexpected error: {type(e).__name__} - {e}",
                    exc_info=True,
                )
                raise

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        try:
            await self._client.close()
            logger.info("QwenLLM client closed.")
        except Exception as e:
            logger.warning(f"Error closing QwenLLM client: {e}")
