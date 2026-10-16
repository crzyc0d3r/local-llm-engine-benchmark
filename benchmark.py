"""Measure one streaming chat endpoint using the Python standard library."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


DEFAULT_PROMPT = "Write a Python function that merges two sorted lists."


def sse_events(lines: Iterable[bytes]) -> Iterator[dict]:
    """Decode SSE data records; control records and comments carry no chat data."""
    data: list[str] = []

    def decode() -> dict | None:
        raw = "\n".join(data)
        if raw == "[DONE]":
            return None
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object in a stream event")
        return value

    for line in lines:
        value = line.decode("utf-8").rstrip("\r\n")
        if not value:
            if data:
                if "\n".join(data) == "[DONE]":
                    return
                event = decode()
                data.clear()
                if event is not None:
                    yield event
        elif value.startswith("data:"):
            data.append(value[5:].removeprefix(" "))
    if data and "\n".join(data) != "[DONE]":
        event = decode()
        if event is not None:
            yield event


@dataclass(frozen=True)
class Measurement:
    engine: str
    endpoint: str
    model: str
    mode: str
    ttft_seconds: float
    end_to_end_seconds: float
    active_stream_seconds: float
    completion_tokens: int | None
    end_to_end_tokens_per_second: float | None
    approximate_decode_tokens_per_second: float | None
    tokens_per_target_pass: float | None
    content_events: int
    text: str
    reasoning: str
    finish_reason: str | None


def positive_int(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


def measure(
    endpoint: str,
    model: str,
    *,
    engine: str = "local",
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 1024,
    timeout: float = 600,
    mode: str = "live",
) -> Measurement:
    """Send one request; token counts are accepted only from server usage data."""
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Endpoint must be a complete http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Endpoint must not contain credentials, query, or fragment")
    if not model or not prompt or max_tokens < 1 or timeout <= 0:
        raise ValueError("Model, prompt, token limit, and timeout must be valid")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    key = os.environ.get("LOCAL_LLM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(endpoint, json.dumps(payload).encode(), headers)
    start = time.perf_counter()
    first = last = None
    text: list[str] = []
    reasoning: list[str] = []
    usage: dict = {}
    events = 0
    finish_reason = None
    actual_mode = mode
    with urlopen(request, timeout=timeout) as response:
        if response.headers.get("X-Synthetic-Fixture", "").lower() == "true":
            actual_mode = "synthetic"
        for event in sse_events(response):
            if event.get("error"):
                raise RuntimeError("Server returned a stream error; inspect its local log")
            if isinstance(event.get("usage"), dict):
                usage.update(event["usage"])
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if content or thought:
                    if not isinstance(content, str) or not isinstance(thought, str):
                        raise ValueError("Expected textual chat stream deltas")
                    last = time.perf_counter()
                    first = last if first is None else first
                    events += 1
                    text.append(content)
                    reasoning.append(thought)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    end = time.perf_counter()
    if first is None or last is None:
        raise RuntimeError("No output arrived; inspect the server log")
    count = positive_int(usage.get("completion_tokens"))
    passes = positive_int(usage.get("spec_verify_ct"))
    span = last - first
    return Measurement(
        engine=engine, endpoint=endpoint, model=model, mode=actual_mode,
        ttft_seconds=first - start, end_to_end_seconds=end - start,
        active_stream_seconds=span, completion_tokens=count,
        end_to_end_tokens_per_second=count / (end - start) if count else None,
        # An SSE content event is not necessarily one token. This is an estimate.
        approximate_decode_tokens_per_second=(count - 1) / span
        if count and count > 1 and span > 0 else None,
        tokens_per_target_pass=count / passes if count and passes else None,
        content_events=events, text="".join(text), reasoning="".join(reasoning),
        finish_reason=finish_reason,
    )


def benchmark(endpoint: str, model: str, *, warmups: int = 1, runs: int = 3,
              **kwargs) -> dict:
    """Discard warmups, retain measured requests and their median metrics."""
    if warmups < 0 or runs < 1:
        raise ValueError("Warmups must be nonnegative and runs positive")
    for _ in range(warmups):
        measure(endpoint, model, **kwargs)
    measurements = [asdict(measure(endpoint, model, **kwargs)) for _ in range(runs)]
    fields = [
        "ttft_seconds", "end_to_end_seconds", "end_to_end_tokens_per_second",
        "approximate_decode_tokens_per_second", "tokens_per_target_pass",
    ]
    medians = {}
    for field in fields:
        values = [m[field] for m in measurements if m[field] is not None]
        medians[field] = statistics.median(values) if values else None
    return {
        "mode": measurements[0]["mode"] if len({m["mode"] for m in measurements}) == 1
        else "mixed", "discarded_warmups": warmups,
        "settings": {"prompt": kwargs.get("prompt", DEFAULT_PROMPT),
                     "temperature": 0, "max_tokens": kwargs.get("max_tokens", 1024),
                     "thinking_requested": False},
        "median": medians, "measurements": measurements,
        "decode_rate_note": "Approximation: (server completion tokens - 1) / span "
                            "from first to last output event. Chunks may contain "
                            "multiple tokens. End-to-end rate uses complete wall time.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get("ENDPOINT"))
    parser.add_argument("--model", default=os.environ.get("MODEL"))
    parser.add_argument("--engine", default="local")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.endpoint or not args.model:
        parser.error("Provide --endpoint and --model (or ENDPOINT and MODEL)")
    if not math.isfinite(args.timeout):
        parser.error("Timeout must be finite")
    try:
        result = benchmark(args.endpoint, args.model, engine=args.engine,
                           prompt=args.prompt, max_tokens=args.max_tokens,
                           timeout=args.timeout, warmups=args.warmups, runs=args.runs)
    except HTTPError as error:
        parser.exit(1, f"HTTP {error.code}; inspect the local server log.\n")
    except (URLError, ValueError, RuntimeError, TimeoutError) as error:
        parser.exit(1, f"Benchmark failed: {type(error).__name__}: {error}\n")
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
