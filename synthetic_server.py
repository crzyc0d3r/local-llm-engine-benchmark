"""A loopback-only synthetic SSE fixture. It performs no model inference."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time


class SyntheticHandler(BaseHTTPRequestHandler):
    first_delay = 0.025
    chunk_delay = 0.012
    include_usage = True
    chunks = ["def merge(a, b):\n", "    return sorted(a + b)", "\n"]
    synthetic_token_count = 16
    requests_seen: list[dict] = []

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 1_000_000:
                raise ValueError("Invalid body length")
            payload = json.loads(self.rfile.read(length))
            if not payload.get("stream"):
                raise ValueError("Expected streaming request")
        except (ValueError, json.JSONDecodeError):
            self.send_error(400)
            return
        self.requests_seen.append(payload)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("X-Synthetic-Fixture", "true")
        self.end_headers()

        def send(event: dict) -> None:
            self.wfile.write(("data: " + json.dumps(event) + "\r\n\r\n").encode())
            self.wfile.flush()

        send({"choices": [{"delta": {"role": "assistant"}}]})
        time.sleep(self.first_delay)
        for index, chunk in enumerate(self.chunks):
            if index:
                time.sleep(self.chunk_delay)
            send({"choices": [{"delta": {"content": chunk}}]})
        send({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        if self.include_usage:
            send({"choices": [], "usage": {
                "completion_tokens": self.synthetic_token_count,
                "spec_verify_ct": 4,
            }})
        self.wfile.write(b"data: [DONE]\r\n\r\n")
        self.wfile.flush()

    def log_message(self, *_args) -> None:
        pass


def create_server(port: int = 0) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), SyntheticHandler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8099)
    options = parser.parse_args()
    with create_server(options.port) as server:
        print(f"SYNTHETIC fixture: http://127.0.0.1:{server.server_port}/v1/chat/completions")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
