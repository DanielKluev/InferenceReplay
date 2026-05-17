"""
Reassembly of streaming SSE responses into non-streaming JSON responses.

Provides functions to parse Server-Sent Events (SSE) chunks from both the
Chat Completions API and the Responses API, and reconstruct complete
non-streaming response objects.

Key functions: `reassemble_streaming_response`, `reassemble_chat_completion`, `reassemble_responses_api`
"""

import json
import logging
from typing import Any

log = logging.getLogger("reassembly")


def _parse_sse_events(chunks: list[str]) -> list[dict[str, Any]]:
    """
    Parse SSE chunks into a list of JSON event data objects.

    Concatenates all chunks (which may contain partial lines or multiple events),
    splits by double-newline boundaries, extracts `data:` lines, and parses each
    as JSON. Skips `data: [DONE]` and non-JSON lines.

    Returns a list of parsed JSON objects from the SSE stream.
    """
    # Concatenate all chunks and split into individual SSE lines
    raw = "".join(chunks)
    events: list[dict[str, Any]] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("data:"):
            # Skip non-data lines (e.g. event: or id: or comments)
            continue

        data_str = line[len("data:"):].strip()
        if data_str == "[DONE]":
            continue

        try:
            events.append(json.loads(data_str))
        except (json.JSONDecodeError, ValueError):
            log.debug("Skipping non-JSON SSE data: %s", data_str[:80])

    return events


def parse_sse_events(chunks: list[str]) -> list[dict[str, Any]]:
    """
    Parse raw SSE transport chunks into complete JSON event objects.

    Public wrapper around the internal parser for storage/replay code that
    needs event-boundary-safe chunks without depending on private helpers.
    """
    return _parse_sse_events(chunks)


def reassemble_chat_completion(chunks: list[str]) -> dict[str, Any]:
    """
    Reassemble streaming Chat Completions SSE chunks into a single non-streaming response.

    Parses all SSE events, merges choice deltas (role, content, tool_calls,
    function_call, refusal), and constructs a complete `chat.completion` object.
    Extracts usage information from the final chunk if present.

    Returns a dict matching the shape of a non-streaming `chat.completion` response.
    """
    events = _parse_sse_events(chunks)
    if not events:
        return {}

    # Use the first event as a template for top-level fields
    first = events[0]
    result: dict[str, Any] = {
        "id": first.get("id", ""),
        "object": "chat.completion",
        "created": first.get("created", 0),
        "model": first.get("model", ""),
        "system_fingerprint": first.get("system_fingerprint"),
    }

    # Accumulate choices from deltas
    choices_map: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] | None = None

    for event in events:
        # Extract usage if present (typically in the final chunk)
        if event.get("usage"):
            usage = event["usage"]

        for choice_delta in event.get("choices", []):
            idx = choice_delta.get("index", 0)

            if idx not in choices_map:
                choices_map[idx] = {
                    "index": idx,
                    "message": {
                        "role": "assistant",
                        "content": None
                    },
                    "finish_reason": None,
                    "logprobs": None,
                }

            choice = choices_map[idx]
            delta = choice_delta.get("delta", {})

            # Update finish_reason
            if choice_delta.get("finish_reason") is not None:
                choice["finish_reason"] = choice_delta["finish_reason"]

            # Accumulate logprobs (each streaming chunk carries one token's logprobs)
            if choice_delta.get("logprobs") is not None:
                delta_logprobs = choice_delta["logprobs"]
                if choice["logprobs"] is None:
                    choice["logprobs"] = {}
                # Accumulate content-level token logprobs
                if delta_logprobs.get("content") is not None:
                    if "content" not in choice["logprobs"]:
                        choice["logprobs"]["content"] = []
                    choice["logprobs"]["content"].extend(delta_logprobs["content"])
                # Accumulate refusal-level token logprobs
                if delta_logprobs.get("refusal") is not None:
                    if "refusal" not in choice["logprobs"]:
                        choice["logprobs"]["refusal"] = []
                    choice["logprobs"]["refusal"].extend(delta_logprobs["refusal"])

            # Merge delta into message
            _merge_delta_into_message(choice["message"], delta)

    # Build sorted choices list
    result["choices"] = [choices_map[idx] for idx in sorted(choices_map.keys())]

    # Attach usage if available
    if usage:
        result["usage"] = usage

    return result


def _merge_delta_into_message(message: dict[str, Any], delta: dict[str, Any]) -> None:
    """
    Merge a streaming delta object into an accumulated message.

    Handles content, role, tool_calls (with argument concatenation),
    function_call, and refusal fields.
    """
    # Role
    if "role" in delta:
        message["role"] = delta["role"]

    # Content
    if "content" in delta and delta["content"] is not None:
        if message.get("content") is None:
            message["content"] = ""
        message["content"] += delta["content"]

    # Reasoning content (used by models that expose chain-of-thought, e.g. DeepSeek-R1, o-series)
    for reasoning_key in ("reasoning_content", "reasoning"):
        if reasoning_key in delta and delta[reasoning_key] is not None:
            if message.get("reasoning_content") is None:
                message["reasoning_content"] = ""
            message["reasoning_content"] += delta[reasoning_key]
            break

    # Refusal
    if "refusal" in delta and delta["refusal"] is not None:
        if message.get("refusal") is None:
            message["refusal"] = ""
        message["refusal"] += delta["refusal"]

    # Tool calls
    if "tool_calls" in delta:
        if "tool_calls" not in message:
            message["tool_calls"] = []
        for tc_delta in delta["tool_calls"]:
            tc_idx = tc_delta.get("index", 0)
            # Extend the list if needed
            while len(message["tool_calls"]) <= tc_idx:
                message["tool_calls"].append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})

            tc = message["tool_calls"][tc_idx]
            if "id" in tc_delta and tc_delta["id"]:
                tc["id"] = tc_delta["id"]
            if "type" in tc_delta:
                tc["type"] = tc_delta["type"]
            if "function" in tc_delta:
                fn_delta = tc_delta["function"]
                if "name" in fn_delta and fn_delta["name"]:
                    tc["function"]["name"] += fn_delta["name"]
                if "arguments" in fn_delta:
                    tc["function"]["arguments"] += fn_delta["arguments"]

    # Legacy function_call
    if "function_call" in delta:
        if "function_call" not in message:
            message["function_call"] = {"name": "", "arguments": ""}
        fc_delta = delta["function_call"]
        if "name" in fc_delta and fc_delta["name"]:
            message["function_call"]["name"] += fc_delta["name"]
        if "arguments" in fc_delta:
            message["function_call"]["arguments"] += fc_delta["arguments"]


def reassemble_responses_api(chunks: list[str]) -> dict[str, Any]:
    """
    Reassemble streaming Responses API SSE chunks into a single non-streaming response.

    Looks for the `response.completed` event which contains the full response object.
    Falls back to reconstructing from available events if `response.completed` is not found.

    Returns a dict matching the shape of a non-streaming Responses API response.
    """
    raw = "".join(chunks)
    lines = raw.splitlines()

    current_event_type: str | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Reset event type on blank line (SSE event boundary)
            current_event_type = None
            continue

        if stripped.startswith("event:"):
            current_event_type = stripped[len("event:"):].strip()
            continue

        if stripped.startswith("data:"):
            data_str = stripped[len("data:"):].strip()
            if data_str == "[DONE]":
                continue

            # The response.completed event contains the full response object
            if current_event_type == "response.completed":
                try:
                    return json.loads(data_str)
                except (json.JSONDecodeError, ValueError):
                    log.warning("Failed to parse response.completed event data")

    # Fallback: if no response.completed event found, try to parse the last valid event
    log.warning("No response.completed event found in Responses API stream, attempting fallback")
    events = _parse_sse_events(chunks)
    if events:
        return events[-1]
    return {}


def reassemble_text_completion(chunks: list[str]) -> dict[str, Any]:
    """
    Reassemble streaming text Completions SSE chunks into a single non-streaming response.

    Handles the ``/v1/completions`` endpoint format where choices carry ``text``
    (concatenated) rather than ``delta.content``. Also preserves ``prompt_logprobs``
    (vLLM-specific, sent in the first chunk) and accumulates per-token ``logprobs``.

    Returns a dict matching the shape of a non-streaming ``text_completion`` response.
    """
    events = _parse_sse_events(chunks)
    if not events:
        return {}

    first = events[0]
    result: dict[str, Any] = {
        "id": first.get("id", ""),
        "object": "text_completion",
        "created": first.get("created", 0),
        "model": first.get("model", ""),
        "system_fingerprint": first.get("system_fingerprint"),
    }

    choices_map: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] | None = None

    for event in events:
        if event.get("usage"):
            usage = event["usage"]

        for choice_delta in event.get("choices", []):
            idx = choice_delta.get("index", 0)

            if idx not in choices_map:
                choices_map[idx] = {
                    "index": idx,
                    "text": "",
                    "finish_reason": None,
                    "logprobs": None,
                    "prompt_logprobs": None,
                }

            choice = choices_map[idx]

            # Concatenate text fragments
            text_part = choice_delta.get("text")
            if text_part is not None:
                choice["text"] += text_part

            # Update finish_reason
            if choice_delta.get("finish_reason") is not None:
                choice["finish_reason"] = choice_delta["finish_reason"]

            # Capture prompt_logprobs (vLLM sends full list in the first chunk)
            if choice_delta.get("prompt_logprobs") is not None:
                choice["prompt_logprobs"] = choice_delta["prompt_logprobs"]

            # Accumulate generation logprobs
            delta_logprobs = choice_delta.get("logprobs")
            if delta_logprobs is not None:
                if choice["logprobs"] is None:
                    choice["logprobs"] = {}
                # OpenAI text completions logprobs format: tokens, token_logprobs, top_logprobs, text_offset
                for key in ("tokens", "token_logprobs", "top_logprobs", "text_offset"):
                    if key in delta_logprobs and delta_logprobs[key] is not None:
                        if key not in choice["logprobs"]:
                            choice["logprobs"][key] = []
                        choice["logprobs"][key].extend(delta_logprobs[key])

    result["choices"] = [choices_map[idx] for idx in sorted(choices_map.keys())]

    if usage:
        result["usage"] = usage

    return result


def reassemble_streaming_response(chunks: list[str], path: str) -> dict[str, Any]:
    """
    Reassemble streaming chunks into a non-streaming response, dispatching by API path.

    Uses `reassemble_chat_completion` for Chat Completions API paths
    (`/v1/chat/completions` or similar), `reassemble_responses_api`
    for Responses API paths (`/v1/responses` or similar), and
    `reassemble_text_completion` for text Completions API paths
    (`/v1/completions` or similar).

    Returns a dict with the reassembled non-streaming response body.
    """
    if "/responses" in path:
        return reassemble_responses_api(chunks)
    if "/completions" in path and "/chat/completions" not in path:
        return reassemble_text_completion(chunks)
    # Default to Chat Completions reassembly
    return reassemble_chat_completion(chunks)
