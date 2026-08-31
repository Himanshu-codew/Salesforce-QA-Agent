"""
Comprehensive tests for QwenLLM.chat_with_tools() and chat().

Tests verify:
1. Successful HTTP 200 response handling (the original initial_e bug)
2. Native tool-call handling
3. Prompt-based fallback tool-call handling
4. Model-not-found (404) auto-discovery
5. Connection errors vs API errors vs programming errors
6. Empty responses, None content, None tool_calls
7. Multiple tool calls
8. Fallback extraction from text
9. Retry logic
"""

import asyncio
import json
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import sys
import os

# Add project root to path so we can import llm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.qwen import (
    QwenLLM,
    LLMConnectionError,
    LLMModelError,
    LLMToolError,
    _extract_text_tool_calls,
    _is_model_not_found,
    _is_connection_error,
    _is_tools_unsupported,
    strip_thinking,
    strip_raw_tool_json,
)
from openai import NotFoundError, APIConnectionError, APIStatusError, RateLimitError


# ---------------------------------------------------------------------------
# Helpers to build mock objects mimicking the OpenAI response structure
# ---------------------------------------------------------------------------

def _make_message(content=None, tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    return msg


def _make_choice(message, finish_reason="stop"):
    c = MagicMock()
    c.message = message
    c.finish_reason = finish_reason
    return c


def _make_response(choices):
    r = MagicMock()
    r.choices = choices
    return r


def _make_native_tool_call(tc_id, name, arguments_dict):
    tc = MagicMock()
    tc.id = tc_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments_dict)
    return tc


def _make_llm():
    """Create a QwenLLM with fully mocked internals (no real HTTP, no .env reload)."""
    with patch("llm.qwen.httpx.AsyncClient"), patch("llm.qwen.AsyncOpenAI"):
        llm = QwenLLM(api_key="test", base_url="http://localhost:11434/v1", model="qwen2.5-coder:14b")
    # Prevent _check_and_update_base_url from loading real .env and recreating _client
    llm._check_and_update_base_url = lambda: None
    return llm


# ============================================================================
# Test Suite
# ============================================================================

class TestStripThinking(unittest.TestCase):
    def test_strips_think_blocks(self):
        self.assertEqual(strip_thinking("<think>reasoning</think> answer"), "answer")

    def test_strips_unclosed_think(self):
        self.assertEqual(strip_thinking("<think>reasoning"), "")

    def test_no_think_block(self):
        self.assertEqual(strip_thinking("hello world"), "hello world")

    def test_empty_string(self):
        self.assertEqual(strip_thinking(""), "")

    def test_none_like(self):
        self.assertEqual(strip_thinking(""), "")


class TestExtractTextToolCalls(unittest.TestCase):
    def test_json_array(self):
        content = json.dumps([{"name": "soqlQuery", "arguments": {"q": "SELECT Id FROM Account"}}])
        calls = _extract_text_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "soqlQuery")
        self.assertIn("q", calls[0]["arguments"])

    def test_json_object(self):
        content = json.dumps({"name": "getUserInfo", "arguments": {}})
        calls = _extract_text_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "getUserInfo")

    def test_markdown_code_block(self):
        content = "Here is the call:\n```json\n[{\"name\": \"soqlQuery\", \"arguments\": {\"q\": \"SELECT Name FROM Account\"}}]\n```"
        calls = _extract_text_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "soqlQuery")

    def test_xml_tool_tags(self):
        content = '<tools>[{"name": "find", "arguments": {"q": "test"}}]</tools>'
        calls = _extract_text_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "find")

    def test_bare_tool_name(self):
        calls = _extract_text_tool_calls("getUserInfo")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "getUserInfo")

    def test_no_tool_calls(self):
        calls = _extract_text_tool_calls("Hello, how can I help you?")
        self.assertEqual(len(calls), 0)

    def test_empty_content(self):
        calls = _extract_text_tool_calls("")
        self.assertEqual(len(calls), 0)

    def test_none_content(self):
        calls = _extract_text_tool_calls(None)
        self.assertEqual(len(calls), 0)

    def test_multiple_tools_in_array(self):
        content = json.dumps([
            {"name": "soqlQuery", "arguments": {"q": "SELECT Id FROM Account"}},
            {"name": "getUserInfo", "arguments": {}},
        ])
        calls = _extract_text_tool_calls(content)
        self.assertEqual(len(calls), 2)

    def test_function_call_syntax(self):
        calls = _extract_text_tool_calls('soqlQuery(q="SELECT Id FROM Account")')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "soqlQuery")


class TestHelperFunctions(unittest.TestCase):
    def test_is_model_not_found_404(self):
        self.assertTrue(_is_model_not_found(Exception("404 model not found")))

    def test_is_model_not_found_not_404(self):
        self.assertFalse(_is_model_not_found(Exception("connection refused")))

    def test_is_connection_error_refused(self):
        self.assertTrue(_is_connection_error(Exception("Connection refused")))

    def test_is_connection_error_timeout(self):
        self.assertTrue(_is_connection_error(Exception("Read timed out")))

    def test_is_connection_error_not_conn(self):
        self.assertFalse(_is_connection_error(Exception("KeyError: 'foo'")))

    def test_is_tools_unsupported(self):
        self.assertTrue(_is_tools_unsupported(Exception("model does not support tools")))

    def test_is_tools_unsupported_no(self):
        self.assertFalse(_is_tools_unsupported(Exception("some other error")))


# ============================================================================
# Core tests for chat_with_tools
# ============================================================================

class TestChatWithTools(unittest.IsolatedAsyncioTestCase):
    """Tests for QwenLLM.chat_with_tools() — no real HTTP calls."""

    # ------------------------------------------------------------------
    # 1. Successful HTTP 200 — the original initial_e BUG scenario
    # ------------------------------------------------------------------
    async def test_success_http200_no_tools(self):
        """
        The ORIGINAL BUG: a successful API call raised UnboundLocalError.
        After fix, a 200 response with no tool calls must return cleanly.
        """
        llm = _make_llm()
        message = _make_message(content="Hello! How can I help?", tool_calls=None)
        choice = _make_choice(message, finish_reason="stop")
        response = _make_response([choice])

        llm._client.chat.completions.create = AsyncMock(return_value=response)

        result = await llm.chat_with_tools(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
        )

        self.assertEqual(result["content"], "Hello! How can I help?")
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(result["finish_reason"], "stop")
        # CRITICAL: must NOT raise UnboundLocalError
        llm._client.chat.completions.create.assert_called_once()

    # ------------------------------------------------------------------
    # 2. Native tool calls
    # ------------------------------------------------------------------
    async def test_native_tool_calls(self):
        llm = _make_llm()
        tc = _make_native_tool_call("tc_1", "soqlQuery", {"q": "SELECT Id FROM Account"})
        message = _make_message(content=None, tool_calls=[tc])
        choice = _make_choice(message, finish_reason="tool_calls")
        response = _make_response([choice])

        llm._client.chat.completions.create = AsyncMock(return_value=response)

        result = await llm.chat_with_tools(
            messages=[{"role": "user", "content": "Show accounts"}],
            tools=[{"type": "function", "function": {"name": "soqlQuery"}}],
        )

        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["name"], "soqlQuery")
        self.assertEqual(result["tool_calls"][0]["id"], "tc_1")
        self.assertEqual(result["tool_calls"][0]["arguments"], {"q": "SELECT Id FROM Account"})
        self.assertEqual(result["finish_reason"], "tool_calls")

    # ------------------------------------------------------------------
    # 3. Multiple native tool calls
    # ------------------------------------------------------------------
    async def test_multiple_native_tool_calls(self):
        llm = _make_llm()
        tc1 = _make_native_tool_call("tc_1", "soqlQuery", {"q": "SELECT Id FROM Account"})
        tc2 = _make_native_tool_call("tc_2", "getUserInfo", {})
        message = _make_message(content=None, tool_calls=[tc1, tc2])
        choice = _make_choice(message, finish_reason="tool_calls")
        response = _make_response([choice])

        llm._client.chat.completions.create = AsyncMock(return_value=response)

        result = await llm.chat_with_tools(
            messages=[{"role": "user", "content": "Show accounts and user info"}],
            tools=[
                {"type": "function", "function": {"name": "soqlQuery"}},
                {"type": "function", "function": {"name": "getUserInfo"}},
            ],
        )

        self.assertEqual(len(result["tool_calls"]), 2)
        names = [tc["name"] for tc in result["tool_calls"]]
        self.assertIn("soqlQuery", names)
        self.assertIn("getUserInfo", names)

    # ------------------------------------------------------------------
    # 4. Text-embedded tool calls (model returns JSON in text, not native)
    # ------------------------------------------------------------------
    async def test_text_embedded_tool_calls(self):
        llm = _make_llm()
        tool_json = json.dumps([{"name": "soqlQuery", "arguments": {"q": "SELECT Id FROM Account"}}])
        message = _make_message(content=tool_json, tool_calls=None)
        choice = _make_choice(message, finish_reason="stop")
        response = _make_response([choice])

        llm._client.chat.completions.create = AsyncMock(return_value=response)

        result = await llm.chat_with_tools(
            messages=[{"role": "user", "content": "Show accounts"}],
            tools=[{"type": "function", "function": {"name": "soqlQuery"}}],
        )

        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["name"], "soqlQuery")

    # ------------------------------------------------------------------
    # 5. Fallback: model doesn't support native tools
    # ------------------------------------------------------------------
    async def test_fallback_tools_unsupported(self):
        """When server rejects native tools, fallback injects schema into prompt."""
        llm = _make_llm()

        tool_json = json.dumps([{"name": "soqlQuery", "arguments": {"q": "SELECT Id FROM Account"}}])
        fallback_msg = _make_message(content=tool_json, tool_calls=None)
        fallback_choice = _make_choice(fallback_msg, finish_reason="stop")
        fallback_response = _make_response([fallback_choice])

        tools_unsupported_err = Exception("model does not support tools")
        mock_create = AsyncMock(side_effect=[tools_unsupported_err, fallback_response])
        llm._client.chat.completions.create = mock_create

        result = await llm.chat_with_tools(
            messages=[{"role": "user", "content": "Show accounts"}],
            tools=[{"type": "function", "function": {"name": "soqlQuery"}}],
        )

        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["name"], "soqlQuery")
        # Verify fallback was called (2 API calls total)
        self.assertEqual(mock_create.call_count, 2)

    # ------------------------------------------------------------------
    # 6. Fallback returns natural language (no tool calls extracted)
    # ------------------------------------------------------------------
    async def test_fallback_natural_language(self):
        """Fallback path where model just answers naturally."""
        llm = _make_llm()

        natural_response = _make_message(content="Sure! Here are your accounts.", tool_calls=None)
        natural_choice = _make_choice(natural_response, finish_reason="stop")
        natural_resp = _make_response([natural_choice])

        tools_err = Exception("model does not support tools")
        mock_create = AsyncMock(side_effect=[tools_err, natural_resp])
        llm._client.chat.completions.create = mock_create

        result = await llm.chat_with_tools(
            messages=[{"role": "user", "content": "Show accounts"}],
            tools=[{"type": "function", "function": {"name": "soqlQuery"}}],
        )

        # The response text "Sure! Here are your accounts." does NOT contain
        # known tool names, so text extraction finds nothing
        self.assertEqual(result["tool_calls"], [])
        self.assertIn("accounts", result["content"])

    # ------------------------------------------------------------------
    # 7. Empty choices
    # ------------------------------------------------------------------
    async def test_empty_choices_raises(self):
        llm = _make_llm()
        response = _make_response([])
        llm._client.chat.completions.create = AsyncMock(return_value=response)

        with self.assertRaises(LLMModelError):
            await llm.chat_with_tools(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
            )

    # ------------------------------------------------------------------
    # 8. None message content (should be converted to "")
    # ------------------------------------------------------------------
    async def test_none_content_becomes_empty_string(self):
        llm = _make_llm()
        message = _make_message(content=None, tool_calls=None)
        choice = _make_choice(message, finish_reason="stop")
        response = _make_response([choice])

        llm._client.chat.completions.create = AsyncMock(return_value=response)

        result = await llm.chat_with_tools(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
        )

        self.assertEqual(result["content"], "")

    # ------------------------------------------------------------------
    # 9. Model not found (404) triggers auto-discovery
    # ------------------------------------------------------------------
    async def test_404_triggers_auto_discovery(self):
        llm = _make_llm()

        not_found_err = NotFoundError(
            message="model not found",
            response=MagicMock(status_code=404, headers={}),
            body=None,
        )

        success_msg = _make_message(content="OK", tool_calls=None)
        success_choice = _make_choice(success_msg, finish_reason="stop")
        success_resp = _make_response([success_choice])

        mock_create = AsyncMock(side_effect=[not_found_err, success_resp])
        llm._client.chat.completions.create = mock_create

        # Mock auto-discovery to return a model
        mock_model = MagicMock()
        mock_model.id = "alternative-model"
        mock_models_resp = MagicMock()
        mock_models_resp.data = [mock_model]
        llm._client.models.list = AsyncMock(return_value=mock_models_resp)

        result = await llm.chat_with_tools(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
        )

        self.assertEqual(result["content"], "OK")
        # Model should have been auto-switched
        self.assertEqual(llm.model, "alternative-model")

    # ------------------------------------------------------------------
    # 10. Connection error — classified as LLMConnectionError
    # ------------------------------------------------------------------
    async def test_connection_error_raises_llm_connection_error(self):
        llm = _make_llm()
        llm._client.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError(request=MagicMock())
        )

        with self.assertRaises(LLMConnectionError) as ctx:
            await llm.chat_with_tools(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
            )
        self.assertIn("Failed to connect to LLM", str(ctx.exception))

    # ------------------------------------------------------------------
    # 11. API status error — classified as LLMModelError
    # ------------------------------------------------------------------
    async def test_api_status_error_raises_llm_model_error(self):
        llm = _make_llm()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.headers = {}
        llm._client.chat.completions.create = AsyncMock(
            side_effect=APIStatusError(
                message="Internal Server Error",
                response=mock_resp,
                body=None,
            )
        )

        with self.assertRaises(LLMModelError) as ctx:
            await llm.chat_with_tools(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
            )
        self.assertIn("HTTP 500", str(ctx.exception))

    # ------------------------------------------------------------------
    # 12. Programming error — NEVER wrapped as connection error
    # ------------------------------------------------------------------
    async def test_programming_error_not_wrapped(self):
        """
        A KeyError from bad internal code must NOT be
        reported as 'Failed to connect to LLM'.
        """
        llm = _make_llm()

        async def raise_key_error(*args, **kwargs):
            raise KeyError("internal_bug_key")

        llm._client.chat.completions.create = AsyncMock(side_effect=raise_key_error)

        with self.assertRaises(KeyError) as ctx:
            await llm.chat_with_tools(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
            )
        # MUST NOT be LLMConnectionError
        self.assertNotIsInstance(ctx.exception, LLMConnectionError)

    # ------------------------------------------------------------------
    # 13. UnboundLocalError scenario (original bug) — must NOT occur
    # ------------------------------------------------------------------
    async def test_successful_response_no_unbound_error(self):
        """
        Directly verifies the original bug scenario:
        POST /v1/chat/completions -> HTTP 200 OK -> should NOT raise UnboundLocalError
        """
        llm = _make_llm()
        message = _make_message(content="Success!", tool_calls=None)
        choice = _make_choice(message, finish_reason="stop")
        response = _make_response([choice])

        llm._client.chat.completions.create = AsyncMock(return_value=response)

        # This call MUST succeed. The original code would raise:
        # UnboundLocalError: cannot access local variable 'initial_e'
        result = await llm.chat_with_tools(
            messages=[{"role": "system", "content": "You are helpful."},
                     {"role": "user", "content": "Hello"}],
            tools=[{"type": "function", "function": {"name": "soqlQuery"}}],
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(result["content"], "Success!")
        self.assertEqual(result["tool_calls"], [])

    # ------------------------------------------------------------------
    # 14. message is None
    # ------------------------------------------------------------------
    async def test_none_message_raises_model_error(self):
        llm = _make_llm()
        choice = MagicMock()
        choice.message = None
        choice.finish_reason = "stop"
        response = _make_response([choice])

        llm._client.chat.completions.create = AsyncMock(return_value=response)

        with self.assertRaises(LLMModelError) as ctx:
            await llm.chat_with_tools(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
            )
        self.assertIn("Message object is None", str(ctx.exception))

    # ------------------------------------------------------------------
    # 15. Tool call with non-JSON arguments (graceful handling)
    # ------------------------------------------------------------------
    async def test_tool_call_bad_arguments(self):
        llm = _make_llm()
        tc = MagicMock()
        tc.id = "tc_1"
        tc.function = MagicMock()
        tc.function.name = "soqlQuery"
        tc.function.arguments = "this is not json"
        message = _make_message(content=None, tool_calls=[tc])
        choice = _make_choice(message, finish_reason="tool_calls")
        response = _make_response([choice])

        llm._client.chat.completions.create = AsyncMock(return_value=response)

        result = await llm.chat_with_tools(
            messages=[{"role": "user", "content": "Query"}],
            tools=[{"type": "function", "function": {"name": "soqlQuery"}}],
        )

        # Should not crash — arguments passed as raw string
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["arguments"], "this is not json")

    # ------------------------------------------------------------------
    # 16. Chat method success path
    # ------------------------------------------------------------------
    async def test_chat_success(self):
        llm = _make_llm()
        message = _make_message(content="<think>thinking</think> Final answer", tool_calls=None)
        choice = _make_choice(message, finish_reason="stop")
        response = _make_response([choice])
        llm._client.chat.completions.create = AsyncMock(return_value=response)

        result = await llm.chat(
            messages=[{"role": "user", "content": "Hello"}],
        )

        self.assertEqual(result, "Final answer")

    # ------------------------------------------------------------------
    # 17. Chat method empty choices
    # ------------------------------------------------------------------
    async def test_chat_empty_choices(self):
        llm = _make_llm()
        response = _make_response([])
        llm._client.chat.completions.create = AsyncMock(return_value=response)

        with self.assertRaises(LLMModelError):
            await llm.chat(messages=[{"role": "user", "content": "Hi"}])

    # ------------------------------------------------------------------
    # 18. Chat method connection error
    # ------------------------------------------------------------------
    async def test_chat_connection_error(self):
        llm = _make_llm()
        llm._client.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError(request=MagicMock())
        )

        with self.assertRaises(LLMConnectionError):
            await llm.chat(messages=[{"role": "user", "content": "Hi"}])

    # ------------------------------------------------------------------
    # 19. Chat method programming error NOT wrapped
    # ------------------------------------------------------------------
    async def test_chat_programming_error_not_wrapped(self):
        llm = _make_llm()

        async def raise_type_error(*args, **kwargs):
            raise TypeError("bad_type")

        llm._client.chat.completions.create = AsyncMock(side_effect=raise_type_error)

        with self.assertRaises(TypeError):
            await llm.chat(messages=[{"role": "user", "content": "Hi"}])

    # ------------------------------------------------------------------
    # 20. Retry on 404 then succeed
    # ------------------------------------------------------------------
    async def test_retry_on_404_then_success(self):
        llm = _make_llm()

        not_found = NotFoundError(
            message="model not found",
            response=MagicMock(status_code=404, headers={}),
            body=None,
        )
        success_msg = _make_message(content="OK", tool_calls=None)
        success_choice = _make_choice(success_msg, finish_reason="stop")
        success_resp = _make_response([success_choice])

        mock_create = AsyncMock(side_effect=[not_found, success_resp])
        llm._client.chat.completions.create = mock_create

        mock_model = MagicMock()
        mock_model.id = "qwen-available"
        llm._client.models.list = AsyncMock(return_value=MagicMock(data=[mock_model]))

        result = await llm.chat_with_tools(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
        )
        self.assertEqual(result["content"], "OK")

    # ------------------------------------------------------------------
    # 21. Non-tools-unsupported error from native call propagates correctly
    # ------------------------------------------------------------------
    async def test_non_tools_error_propagates(self):
        """When native tools fail for a reason OTHER than 'not supported',
        the error should NOT trigger the prompt fallback."""
        llm = _make_llm()

        # Simulate a generic API error (not tools-unsupported)
        mock_create = AsyncMock(side_effect=RuntimeError("server overloaded"))
        llm._client.chat.completions.create = mock_create

        with self.assertRaises(RuntimeError) as ctx:
            await llm.chat_with_tools(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[{"type": "function", "function": {"name": "soqlQuery"}}],
            )
        # Should be called only once (no fallback attempt)
        self.assertEqual(mock_create.call_count, 1)
        self.assertNotIsInstance(ctx.exception, LLMConnectionError)

    # ------------------------------------------------------------------
    # 22. Content sanitization strips tool JSON from response text
    # ------------------------------------------------------------------
    async def test_content_sanitization_strips_tool_json(self):
        """Response text containing embedded tool JSON should be cleaned."""
        llm = _make_llm()
        content_with_json = (
            "Let me query that.\n"
            "```json\n[{\"name\": \"soqlQuery\", \"arguments\": {\"q\": \"SELECT Id FROM Account\"}}]\n```\n"
            "Here are your results."
        )
        message = _make_message(content=content_with_json, tool_calls=None)
        choice = _make_choice(message, finish_reason="stop")
        response = _make_response([choice])

        llm._client.chat.completions.create = AsyncMock(return_value=response)

        result = await llm.chat_with_tools(
            messages=[{"role": "user", "content": "Show accounts"}],
            tools=[{"type": "function", "function": {"name": "soqlQuery"}}],
        )

        # Should have extracted tool call from text
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["name"], "soqlQuery")
        # Content should have tool JSON stripped
        self.assertNotIn("soqlQuery", result["content"])

    # ------------------------------------------------------------------
    # 23. Rate limit error is NOT treated as connection error
    # ------------------------------------------------------------------
    async def test_rate_limit_error(self):
        llm = _make_llm()
        llm._client.chat.completions.create = AsyncMock(
            side_effect=RateLimitError(
                message="rate limit exceeded",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            )
        )

        # RateLimitError inherits from APIStatusError, so should become LLMModelError
        with self.assertRaises((LLMModelError, RateLimitError)):
            await llm.chat_with_tools(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
            )

    # ------------------------------------------------------------------
    # 24. Verify tool_calls property truthiness for MagicMock list
    # ------------------------------------------------------------------
    async def test_tool_calls_list_iteration(self):
        """Ensure native tool_calls list iteration works correctly."""
        llm = _make_llm()
        tc = _make_native_tool_call("tc_1", "getUserInfo", {})
        message = _make_message(content="", tool_calls=[tc])
        choice = _make_choice(message, finish_reason="tool_calls")
        response = _make_response([choice])

        llm._client.chat.completions.create = AsyncMock(return_value=response)

        result = await llm.chat_with_tools(
            messages=[{"role": "user", "content": "Who am I?"}],
            tools=[{"type": "function", "function": {"name": "getUserInfo"}}],
        )

        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["name"], "getUserInfo")
        self.assertEqual(result["tool_calls"][0]["id"], "tc_1")


# ============================================================================
# Verify the _extract_text_tool_calls function works with nested structures
# ============================================================================

class TestExtractTextToolCallsAdvanced(unittest.TestCase):
    def test_arguments_as_string(self):
        """Model returns arguments as a stringified JSON."""
        content = json.dumps({"name": "soqlQuery", "arguments": '{"q": "SELECT Id FROM Account"}'})
        calls = _extract_text_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"], {"q": "SELECT Id FROM Account"})

    def test_function_key_wrapping(self):
        content = json.dumps({"function": "getUserInfo", "arguments": {}})
        calls = _extract_text_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "getUserInfo")

    def test_mixed_content_with_json(self):
        content = "Let me query that for you.\n```json\n[{\"name\": \"soqlQuery\", \"arguments\": {\"q\": \"SELECT Name FROM Lead\"}}]\n```"
        calls = _extract_text_tool_calls(content)
        self.assertEqual(len(calls), 1)

    def test_smart_bracket_extraction(self):
        content = "Here is the result: [{\"name\": \"find\", \"arguments\": {\"q\": \"test\"}}]"
        calls = _extract_text_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "find")


if __name__ == "__main__":
    unittest.main()
