"""The only module that talks to the Claude API, and the only one that imports `anthropic`.

Everything else sees a `Transport`: plain dicts in, plain dicts out. Tests use
`FakeTransport`, which never opens a socket; `AnthropicTransport` wraps the
pinned SDK and is created only by the live commands, after the ledger has
admitted the spend.

The SDK's own retries are switched off (`max_retries=0`) so that every attempt
passes through the ledger. What a failure means for the money:

- an HTTP error response (4xx, 5xx) was not billed: the attempt settles at $0;
- a dropped connection or a timeout may have been billed: the attempt stays
  spent at its bound ("unknown").

The key lives in the macOS Keychain (service `raglab-stage4`), never in the
environment: a key exported in a shell can be picked up and billed by other
tools, Claude Code included, outside this stage's ledger. It is read only when
a live client is made, handed to the SDK as an explicit argument (so the SDK
consults no credential variables, profiles or federation settings) and never
kept, logged or written. The client refuses to start if the environment holds
a key or any other Anthropic setting that could redirect, re-authenticate,
decorate or log its requests, or if the installed SDK isn't the pinned one.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Callable, Iterator, Mapping, Protocol, Sequence

OFFICIAL_BASE_URL = "https://api.anthropic.com"
REQUEST_ID = "_request_id"  # where create() puts the response's request-id header
# The body keys the token-counting endpoint takes (max_tokens is not one: counting is about input).
COUNT_KEYS = ("model", "system", "messages", "thinking", "output_config", "tools", "tool_choice", "cache_control")


class LlmError(RuntimeError):
    """The API can't be used as asked; nothing was sent."""


@dataclass
class ApiError(Exception):
    """A failed attempt, classified by what it means for the bill.

    kind: "client" (4xx other than 429), "rate_limited" (429), "server" (5xx) are HTTP
    responses, never billed; "connection" (no response) may have been billed.
    """

    kind: str
    status: int | None = None
    retry_after: float | None = None
    request_id: str | None = None
    message: str = ""

    def __str__(self) -> str:
        where = f" (HTTP {self.status})" if self.status else ""
        rid = f" [{self.request_id}]" if self.request_id else ""
        return f"{self.kind}{where}{rid}: {self.message}"

    @property
    def billed_possibly(self) -> bool:
        return self.kind == "connection"


KEYCHAIN_SERVICE = "raglab-stage4"
PINNED_SDK = "2.6.0"


def check_environment(env: Mapping[str, str] | None = None) -> None:
    """Refuse an environment that could spend, send or log outside this stage's control. Never shows a value."""
    env = os.environ if env is None else env
    if env.get("ANTHROPIC_API_KEY"):
        raise LlmError("ANTHROPIC_API_KEY is set in this shell. Remove it from your shell startup files and keep "
                       f"the key in the Keychain (service {KEYCHAIN_SERVICE!r}): a key in the environment can be "
                       "billed by other tools, outside the ledger.")
    others = sorted(k for k in env if k.startswith("ANTHROPIC_") and k != "ANTHROPIC_BASE_URL")
    if others:
        raise LlmError(f"unset {', '.join(others)}: they can change where requests go, how they authenticate, "
                       "what headers they carry or whether they are logged.")
    base = env.get("ANTHROPIC_BASE_URL")
    if base and base.rstrip("/") != OFFICIAL_BASE_URL:
        raise LlmError(f"ANTHROPIC_BASE_URL points somewhere other than {OFFICIAL_BASE_URL}; unset it.")


KEYCHAIN_HELP = (f"Store it in your own terminal, trusting no app, so every read asks you first:\n"
                 f'  security add-generic-password -a "$USER" -s {KEYCHAIN_SERVICE} -T "" -w')


def keychain_key(service: str = KEYCHAIN_SERVICE, run: Callable = subprocess.run) -> str:
    """The API key from the macOS Keychain. Errors name the item, never its contents.

    The item is created trusting no application (-T ""), so each read raises a
    macOS dialog the developer must answer: that dialog, not the typed phrase,
    is the human gate on every billed command. Allow, never Always Allow.
    """
    try:
        found = run(["security", "find-generic-password", "-s", service, "-w"], capture_output=True, text=True,
                    timeout=180)
    except (OSError, subprocess.SubprocessError):
        raise LlmError("couldn't read the Keychain (no answer to its dialog?)") from None
    key = (found.stdout or "").strip() if found.returncode == 0 else ""
    if not key:
        raise LlmError(f"no API key in the Keychain under {service!r}, or access was denied. {KEYCHAIN_HELP}")
    return key


def check_sdk(sdk) -> None:
    """The installed SDK must be the pinned one and send the pinned API version."""
    from raglab.store import API_VERSION

    version = getattr(sdk, "__version__", None)
    if version != PINNED_SDK:
        raise LlmError(f"anthropic {version} is installed; Stage 4 is pinned to {PINNED_SDK} (requirements-lock.txt)")
    sent = getattr(getattr(sdk, "_client", None), "ANTHROPIC_API_VERSION", None)
    if sent != API_VERSION:
        raise LlmError(f"the SDK sends API version {sent}; the plan's requests are keyed to {API_VERSION}")


class Transport(Protocol):
    def create(self, body: Mapping) -> dict: ...
    def stream(self, body: Mapping) -> dict: ...
    def count_tokens(self, body: Mapping) -> int: ...
    def batch_create(self, requests: Sequence[Mapping]) -> dict: ...
    def batch_retrieve(self, batch_id: str) -> dict: ...
    def batch_results(self, batch_id: str) -> Iterator[dict]: ...
    def batch_cancel(self, batch_id: str) -> dict: ...
    def batch_list(self) -> Iterator[dict]: ...  # newest first, every page


def count_body(body: Mapping) -> dict:
    return {k: body[k] for k in COUNT_KEYS if k in body}


class AnthropicTransport:
    """The real API, through the pinned SDK. Created only by live commands."""

    def __init__(self, key: str, env: Mapping[str, str] | None = None, timeout: float = 600.0) -> None:
        check_environment(env)
        import anthropic  # imported here so nothing else in the package can reach the network

        from raglab.store import API_VERSION

        check_sdk(anthropic)
        self._sdk = anthropic
        self._client = anthropic.Anthropic(api_key=key, base_url=OFFICIAL_BASE_URL, max_retries=0,
                                           timeout=timeout, default_headers={"anthropic-version": API_VERSION})

    @staticmethod
    def _json(obj) -> dict:
        """The SDK object as plain JSON types: datetimes become ISO strings, so records are canonical JSON."""
        return obj.to_dict(mode="json")

    def _call(self, fn: Callable, *args, **kwargs):
        a = self._sdk
        try:
            return fn(*args, **kwargs)
        except a.APIConnectionError as e:  # includes timeouts: no response, so it may have been processed
            raise ApiError("connection", message=type(e).__name__) from None
        except a.RateLimitError as e:
            retry = e.response.headers.get("retry-after") if getattr(e, "response", None) is not None else None
            raise ApiError("rate_limited", 429, float(retry) if retry else None, getattr(e, "request_id", None),
                           getattr(e, "message", "")[:300]) from None
        except a.APIStatusError as e:
            kind = "server" if e.status_code >= 500 else "client"
            raise ApiError(kind, e.status_code, None, getattr(e, "request_id", None),
                           getattr(e, "message", "")[:300]) from None

    def create(self, body: Mapping) -> dict:
        message = self._call(self._client.messages.create, **body)
        out = self._json(message)
        out[REQUEST_ID] = getattr(message, "_request_id", None)  # kept beside the message, never sent anywhere
        return out

    def stream(self, body: Mapping) -> dict:
        """A streamed request, timed: the final message, seconds to the first text, and seconds in all.

        A failure before the response starts is classified as usual. Once the
        stream is open, input and streamed output may already be billed, so any
        failure after that is "connection": the attempt stays spent at its bound.
        """
        start = time.monotonic()
        manager = self._call(self._client.messages.stream, **body)
        try:
            stream = self._call(manager.__enter__)
        except ApiError:
            raise
        first = None
        try:
            for event in stream:
                if first is None and getattr(event, "type", "") == "content_block_delta" and \
                        getattr(getattr(event, "delta", None), "type", "") == "text_delta":
                    first = time.monotonic() - start
            message = stream.get_final_message()
        except Exception as e:  # noqa: BLE001 - after the stream opened, every failure may have been billed
            raise ApiError("connection", message=f"stream broke after it opened ({type(e).__name__})") from None
        finally:
            manager.__exit__(None, None, None)
        out = self._json(message)
        out[REQUEST_ID] = getattr(stream, "request_id", None)
        return {"message": out, "first_text_seconds": first, "total_seconds": time.monotonic() - start}

    def count_tokens(self, body: Mapping) -> int:
        return int(self._call(self._client.messages.count_tokens, **count_body(body)).input_tokens)

    def batch_create(self, requests: Sequence[Mapping]) -> dict:
        reqs = [{"custom_id": r["custom_id"], "params": dict(r["params"])} for r in requests]
        return self._json(self._call(self._client.messages.batches.create, requests=reqs))

    def batch_retrieve(self, batch_id: str) -> dict:
        return self._json(self._call(self._client.messages.batches.retrieve, batch_id))

    def batch_results(self, batch_id: str) -> Iterator[dict]:
        items = self._call(self._client.messages.batches.results, batch_id)
        while True:  # results stream in; a dropped stream mid-way is classified like any other failure
            try:
                item = self._call(next, items)
            except StopIteration:
                return
            yield self._json(item)

    def batch_cancel(self, batch_id: str) -> dict:
        return self._json(self._call(self._client.messages.batches.cancel, batch_id))

    def batch_list(self) -> Iterator[dict]:
        """Every batch in the workspace, newest first; the SDK pages on as the caller reads."""
        batches = iter(self._call(self._client.messages.batches.list, limit=100))
        while True:
            try:
                batch = self._call(next, batches)
            except StopIteration:
                return
            yield self._json(batch)


# ---------------------------------------------------------------------------------------------------- the fake

Responder = Callable[[Mapping], dict]


@dataclass
class FakeTransport:
    """A stand-in API for tests: a responder function plays the model, nothing touches the network.

    `responder(body)` returns a Message dict, or raises ApiError to simulate a
    failure. Batch items can be made to error or expire through `batch_outcome`.
    Every call is recorded in `calls`.
    """

    responder: Responder
    count: Callable[[Mapping], int] = lambda body: 1000
    batch_outcome: Callable[[str], str] = lambda custom_id: "succeeded"
    polls_until_ended: int = 1
    calls: list[tuple[str, object]] = field(default_factory=list)
    _batches: dict[str, dict] = field(default_factory=dict)
    _polls: dict[str, int] = field(default_factory=dict)

    def create(self, body: Mapping) -> dict:
        self.calls.append(("create", body))
        return self.responder(body)

    def stream(self, body: Mapping) -> dict:
        self.calls.append(("stream", body))
        return {"message": self.responder(body), "first_text_seconds": 0.1, "total_seconds": 0.5}

    def count_tokens(self, body: Mapping) -> int:
        self.calls.append(("count_tokens", count_body(body)))
        return self.count(body)

    def batch_create(self, requests: Sequence[Mapping]) -> dict:
        batch_id = f"msgbatch_fake{len(self._batches) + 1:04d}"
        self.calls.append(("batch_create", [r["custom_id"] for r in requests]))
        self._batches[batch_id] = {"requests": [dict(r) for r in requests], "status": "in_progress"}
        self._polls[batch_id] = 0
        created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self._batches[batch_id]["created_at"] = created
        return {"id": batch_id, "processing_status": "in_progress", "type": "message_batch", "created_at": created}

    def batch_retrieve(self, batch_id: str) -> dict:
        self.calls.append(("batch_retrieve", batch_id))
        b = self._batches[batch_id]
        self._polls[batch_id] += 1
        if b["status"] == "in_progress" and self._polls[batch_id] >= self.polls_until_ended:
            b["status"] = "ended"
        return {"id": batch_id, "processing_status": b["status"], "type": "message_batch"}

    def batch_results(self, batch_id: str) -> Iterator[dict]:
        self.calls.append(("batch_results", batch_id))
        b = self._batches[batch_id]
        if b["status"] != "ended":
            raise ApiError("client", 409, message="batch has not ended")
        for r in reversed(b["requests"]):  # results come back in any order
            outcome = "canceled" if b.get("canceled") else self.batch_outcome(r["custom_id"])
            if outcome == "succeeded":
                yield {"custom_id": r["custom_id"], "result": {"type": "succeeded", "message": self.responder(r["params"])}}
            elif outcome == "errored":
                yield {"custom_id": r["custom_id"],
                       "result": {"type": "errored", "error": {"type": "error",
                                                               "error": {"type": "api_error", "message": "boom"}}}}
            else:
                yield {"custom_id": r["custom_id"], "result": {"type": outcome}}

    def batch_cancel(self, batch_id: str) -> dict:
        self.calls.append(("batch_cancel", batch_id))
        self._batches[batch_id]["canceled"] = True
        self._batches[batch_id]["status"] = "ended"
        return {"id": batch_id, "processing_status": "canceling"}

    def batch_list(self) -> Iterator[dict]:
        self.calls.append(("batch_list", None))
        for i, b in reversed(list(self._batches.items())):
            yield {"id": i, "processing_status": b["status"], "created_at": b["created_at"]}


def fake_message(text: str = "An answer.", *, model: str = "claude-haiku-5-5", input_tokens: int = 1000,
                 output_tokens: int = 200, stop_reason: str = "end_turn", content: list | None = None,
                 cache_read: int = 0, cache_write: int = 0, stop_details: Mapping | None = None) -> dict:
    """A Message dict shaped like the API's, for fixtures."""
    return {
        "id": "msg_fake", "type": "message", "role": "assistant", "model": model,
        "content": content if content is not None else [{"type": "text", "text": text}],
        "stop_reason": stop_reason, "stop_sequence": None, "stop_details": dict(stop_details) if stop_details else None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                  "cache_creation_input_tokens": cache_write, "cache_read_input_tokens": cache_read},
    }
