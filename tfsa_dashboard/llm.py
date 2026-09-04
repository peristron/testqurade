"""Provider-neutral chat clients and a bounded research tool loop."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

from .config import ProviderSettings
from .errors import DashboardError, ProviderError


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Completion:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    text: str
    tool_events: list[dict[str, Any]]


class ToolExecutor(Protocol):
    def execute(self, name: str, arguments: dict[str, Any]) -> Any: ...


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProviderError("The model returned invalid tool arguments.") from exc
    if not isinstance(parsed, dict):
        raise ProviderError("The model returned non-object tool arguments.")
    return parsed


def _message_content_as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str)


def _strict_alternating_messages(
    messages: list[dict[str, Any]], extra_system_content: str
) -> list[dict[str, Any]]:
    """Prepare Gemma-style history with one system message and alternating turns."""
    system_parts: list[str] = []
    conversation: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = _message_content_as_text(message.get("content", ""))
        if role == "system":
            if content.strip():
                system_parts.append(content.strip())
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        if conversation and conversation[-1]["role"] == role:
            conversation[-1]["content"] = (
                f"{conversation[-1]['content']}\n\n{content}".strip()
            )
        else:
            conversation.append({"role": role, "content": content})

    if extra_system_content.strip():
        system_parts.append(extra_system_content.strip())
    prepared: list[dict[str, Any]] = []
    if system_parts:
        prepared.append({"role": "system", "content": "\n\n".join(system_parts)})
    prepared.extend(conversation)
    return prepared


class BaseLLMClient(ABC):
    def __init__(self, settings: ProviderSettings, timeout: float = 45) -> None:
        self.settings = settings
        self.timeout = timeout

    @abstractmethod
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> Completion:
        raise NotImplementedError

    @abstractmethod
    def tool_result_message(self, call_id: str, result: Any) -> dict[str, Any]:
        raise NotImplementedError

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response: requests.Response | None = None
        for attempt in range(2):
            try:
                response = requests.post(
                    f"{self.settings.base_url}/{path.lstrip('/')}",
                    headers={
                        "Authorization": f"Bearer {self.settings.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise ProviderError(f"Could not reach {self.settings.label}.") from exc
            if response.status_code != 429 or attempt == 1:
                break
            retry_header = response.headers.get("Retry-After", "")
            try:
                retry_delay = float(retry_header)
            except (TypeError, ValueError):
                retry_delay = 2.0
            time.sleep(max(0.5, min(retry_delay, 5.0)))
        if response is None:
            raise ProviderError(f"Could not reach {self.settings.label}.")
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.settings.label} returned an invalid response.") from exc
        if not response.ok:
            error = data.get("error", data.get("message", response.reason))
            if isinstance(error, dict):
                error = error.get("message", str(error))
            raise ProviderError(
                f"{self.settings.label} request failed ({response.status_code}): {str(error)[:400]}"
            )
        if not isinstance(data, dict):
            raise ProviderError(f"{self.settings.label} returned an unexpected response.")
        return data


class OpenAICompatibleClient(BaseLLMClient):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> Completion:
        request_messages = list(messages)
        if tools and not self.settings.native_tools:
            text_tool_instruction = (
                "You can request only the read-only tools in the JSON schemas below. When a "
                "tool is needed, respond with exactly one JSON object and no Markdown: "
                '{"tool_calls":[{"id":"call_1","name":"tool_name","arguments":{}}]}. '
                "You may request multiple calls. After TOOL_RESULT messages, either request "
                "another tool the same way or answer the user normally. Never invent a tool.\n"
                + json.dumps(tools)
            )
            request_messages = _strict_alternating_messages(
                request_messages, text_tool_instruction
            )
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": request_messages,
            "temperature": 0.2,
            "stream": False,
        }
        if tools and self.settings.native_tools:
            payload.update({"tools": tools, "tool_choice": "auto"})
        data = self._post("chat/completions", payload)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.settings.label} returned no assistant message.") from exc
        content = message.get("content") or ""
        if isinstance(content, list):
            text = "".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict)
            )
        else:
            text = str(content)
        calls = []
        for raw in message.get("tool_calls", []) or []:
            function = raw.get("function", {})
            calls.append(
                ToolCall(
                    call_id=str(raw.get("id", "tool_call")),
                    name=str(function.get("name", "")),
                    arguments=_parse_arguments(function.get("arguments", {})),
                )
            )
        if tools and not self.settings.native_tools and not calls:
            candidate = text.strip()
            if candidate.startswith("```") and candidate.endswith("```"):
                candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            try:
                text_payload = json.loads(candidate)
            except (TypeError, json.JSONDecodeError):
                text_payload = {}
            if not isinstance(text_payload, dict):
                text_payload = {}
            for index, raw in enumerate(text_payload.get("tool_calls", []) or []):
                if not isinstance(raw, dict):
                    continue
                calls.append(
                    ToolCall(
                        call_id=str(raw.get("id", f"text_call_{index}")),
                        name=str(raw.get("name", "")),
                        arguments=_parse_arguments(raw.get("arguments", {})),
                    )
                )
        assistant_message = {"role": "assistant", "content": content}
        if message.get("tool_calls"):
            assistant_message["tool_calls"] = message["tool_calls"]
        return Completion(text=text, tool_calls=calls, assistant_message=assistant_message)

    def tool_result_message(self, call_id: str, result: Any) -> dict[str, Any]:
        if not self.settings.native_tools:
            return {
                "role": "user",
                "content": f"TOOL_RESULT {call_id}: {json.dumps(result, default=str)}",
            }
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result, default=str),
        }


class CohereClient(BaseLLMClient):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = tools
        data = self._post("chat", payload)
        try:
            message = data["message"]
        except (KeyError, TypeError) as exc:
            raise ProviderError("Cohere returned no assistant message.") from exc
        content = message.get("content", []) or []
        text = "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
        calls = []
        for raw in message.get("tool_calls", []) or []:
            function = raw.get("function", {})
            calls.append(
                ToolCall(
                    call_id=str(raw.get("id", "tool_call")),
                    name=str(function.get("name", "")),
                    arguments=_parse_arguments(function.get("arguments", {})),
                )
            )
        assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
        if message.get("tool_calls"):
            assistant_message["tool_calls"] = message["tool_calls"]
        if message.get("tool_plan"):
            assistant_message["tool_plan"] = message["tool_plan"]
        return Completion(text=text, tool_calls=calls, assistant_message=assistant_message)

    def tool_result_message(self, call_id: str, result: Any) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": [
                {
                    "type": "document",
                    "document": {"data": json.dumps(result, default=str)},
                }
            ],
        }


def create_client(settings: ProviderSettings) -> BaseLLMClient:
    if settings.protocol == "cohere":
        return CohereClient(settings)
    return OpenAICompatibleClient(settings)


def run_research_agent(
    client: BaseLLMClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    executor: ToolExecutor,
    max_rounds: int = 4,
) -> AgentResult:
    """Run a bounded tool loop; no tool can place or modify a trade."""
    working = list(messages)
    events: list[dict[str, Any]] = []
    for _ in range(max_rounds):
        completion = client.complete(working, tools)
        if not completion.tool_calls:
            if not completion.text.strip():
                raise ProviderError("The model returned an empty response.")
            return AgentResult(text=completion.text, tool_events=events)
        working.append(completion.assistant_message)
        for call in completion.tool_calls:
            try:
                result = executor.execute(call.name, call.arguments)
                event = {"tool": call.name, "arguments": call.arguments, "result": result}
            except (DashboardError, TypeError, ValueError) as exc:
                result = {"error": str(exc)}
                event = {"tool": call.name, "arguments": call.arguments, "error": str(exc)}
            events.append(event)
            working.append(client.tool_result_message(call.call_id, result))
    raise ProviderError("The research agent reached its tool-call limit. Please narrow the request.")
