"""Model back ends for the evaluation harness: Ollama (local), Gemini (free tier), Anthropic.

Every back end takes the same neutral conversation and returns the same reply, so the
agent loop does not know which model it is driving. Only the standard library is used
for Ollama and Gemini; Anthropic needs the `anthropic` package.

Neutral conversation: a list of dicts
    {"role": "user", "text": str}
    {"role": "assistant", "text": str, "calls": [{"id", "name", "args"}]}
    {"role": "tool", "id": str, "name": str, "output": str}
Reply: {"text": str, "calls": [...], "input_tokens": int, "output_tokens": int, "stop": str}

Keys are read from the environment and never written to disk:
    GEMINI_API_KEY    (Google AI Studio free key)
    ANTHROPIC_API_KEY
Ollama needs no key; the host defaults to http://localhost:11434 (override with OLLAMA_HOST).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

Reply = dict[str, Any]
Messages = list[dict[str, Any]]


def _post(
    url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    """POST JSON; retry rate limits and transient server errors with backoff (free tiers throttle)."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", **headers}
    )
    delay = 5.0
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as err:
            detail = err.read().decode(errors="replace")[:500]
            if err.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(delay)
                delay = min(delay * 2, 90.0)
                continue
            raise RuntimeError(
                f"HTTP {err.code} from {url.split('?', maxsplit=1)[0]}: {detail}"
            ) from err
        except (urllib.error.URLError, TimeoutError) as err:
            if attempt < 5:
                time.sleep(delay)
                delay = min(delay * 2, 90.0)
                continue
            raise RuntimeError(f"cannot reach {url.split('?', maxsplit=1)[0]}: {err}") from err
    raise RuntimeError("unreachable")


class Ollama:
    """A local model served by Ollama, using its tool-calling chat endpoint."""

    def __init__(
        self, model: str, system: str, tool: dict[str, Any], num_ctx: int, timeout: float
    ) -> None:
        self.model, self.system, self.tool = model, system, tool
        self.num_ctx, self.timeout = num_ctx, timeout
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host

    def step(self, messages: Messages, max_tokens: int) -> Reply:
        wire: list[dict[str, Any]] = [{"role": "system", "content": self.system}]
        for m in messages:
            if m["role"] == "user":
                wire.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": m["text"]}
                if m["calls"]:
                    entry["tool_calls"] = [
                        {"function": {"name": c["name"], "arguments": c["args"]}}
                        for c in m["calls"]
                    ]
                wire.append(entry)
            else:
                wire.append({"role": "tool", "tool_name": m["name"], "content": m["output"]})
        body = {
            "model": self.model,
            "messages": wire,
            "tools": [{"type": "function", "function": self.tool}],
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "num_predict": max_tokens, "temperature": 0.2},
        }
        resp = _post(f"{self.host}/api/chat", body, {}, self.timeout)
        msg = resp.get("message", {})
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            args = c["function"].get("arguments") or {}
            if isinstance(args, str):  # some models return the arguments as a JSON string
                args = json.loads(args or "{}")
            calls.append({"id": f"call{i}", "name": c["function"]["name"], "args": args})
        return {
            "text": msg.get("content") or "",
            "calls": calls,
            "input_tokens": int(resp.get("prompt_eval_count", 0)),
            "output_tokens": int(resp.get("eval_count", 0)),
            "stop": str(resp.get("done_reason", "stop")),
        }


class Gemini:
    """Google Gemini through the Generative Language REST API (free-tier key from AI Studio)."""

    def __init__(self, model: str, system: str, tool: dict[str, Any], timeout: float) -> None:
        self.model, self.system, self.tool, self.timeout = model, system, tool, timeout
        self.key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.key:
            raise SystemExit("set GEMINI_API_KEY in the environment (it is never written to disk)")

    def step(self, messages: Messages, max_tokens: int) -> Reply:
        contents: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": m["text"]}]})
            elif m["role"] == "assistant":
                parts: list[dict[str, Any]] = []
                if m["text"]:
                    parts.append({"text": m["text"]})
                parts += [
                    {"functionCall": {"name": c["name"], "args": c["args"]}} for c in m["calls"]
                ]
                contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            else:
                part = {
                    "functionResponse": {"name": m["name"], "response": {"output": m["output"]}}
                }
                if (
                    contents
                    and contents[-1]["role"] == "user"
                    and "functionResponse" in contents[-1]["parts"][0]
                ):
                    contents[-1]["parts"].append(
                        part
                    )  # several results from one turn go in one message
                else:
                    contents.append({"role": "user", "parts": [part]})
        body = {
            "systemInstruction": {"parts": [{"text": self.system}]},
            "contents": contents,
            "tools": [{"functionDeclarations": [self.tool]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2},
        }
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        resp = _post(url, body, {"x-goog-api-key": str(self.key)}, self.timeout)
        cands = resp.get("candidates") or []
        parts = (cands[0].get("content", {}).get("parts") or []) if cands else []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        calls = [
            {
                "id": f"call{i}",
                "name": p["functionCall"]["name"],
                "args": p["functionCall"].get("args") or {},
            }
            for i, p in enumerate(p for p in parts if "functionCall" in p)
        ]
        use = resp.get("usageMetadata", {})
        return {
            "text": text,
            "calls": calls,
            "input_tokens": int(use.get("promptTokenCount", 0)),
            "output_tokens": int(use.get("candidatesTokenCount", 0)),
            "stop": str(cands[0].get("finishReason", "STOP")) if cands else "NO_CANDIDATE",
        }


class Anthropic:
    """Claude models through the `anthropic` package."""

    def __init__(self, model: str, system: str, tool: dict[str, Any]) -> None:
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "set ANTHROPIC_API_KEY in the environment (it is never written to disk)"
            )
        self.client = anthropic.Anthropic()
        self.model, self.system = model, system
        self.tool = {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }

    def step(self, messages: Messages, max_tokens: int) -> Reply:
        wire: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "user":
                wire.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                blocks: list[dict[str, Any]] = (
                    [{"type": "text", "text": m["text"]}] if m["text"] else []
                )
                blocks += [
                    {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                    for c in m["calls"]
                ]
                wire.append({"role": "assistant", "content": blocks})
            else:
                block = {"type": "tool_result", "tool_use_id": m["id"], "content": m["output"]}
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                    wire[-1]["content"].append(block)
                else:
                    wire.append({"role": "user", "content": [block]})
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system,
            tools=[self.tool],
            messages=wire,
        )
        calls = [
            {"id": b.id, "name": b.name, "args": dict(b.input)}
            for b in resp.content
            if b.type == "tool_use"
        ]
        text = "".join(b.text for b in resp.content if b.type == "text")
        return {
            "text": text,
            "calls": calls,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "stop": resp.stop_reason or "end_turn",
        }


def make(
    provider: str, model: str, system: str, tool: dict[str, Any], *, num_ctx: int, timeout: float
) -> Any:
    if provider == "ollama":
        return Ollama(model, system, tool, num_ctx, timeout)
    if provider == "gemini":
        return Gemini(model, system, tool, timeout)
    if provider == "anthropic":
        return Anthropic(model, system, tool)
    raise SystemExit(f"unknown provider {provider!r}")
