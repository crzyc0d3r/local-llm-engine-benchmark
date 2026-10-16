"""Exercise actual HTTP streaming, timing boundaries, and missing usage data."""

from io import BytesIO
import json
from threading import Thread
import unittest

from benchmark import benchmark, measure, sse_events
from synthetic_server import SyntheticHandler, create_server


class BenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        SyntheticHandler.requests_seen = []
        SyntheticHandler.include_usage = True
        self.server = create_server()
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def test_actual_stream_timing_and_payload(self) -> None:
        result = measure(self.endpoint, "fixture", mode="synthetic")
        # The role-only event must not become the first output timestamp.
        self.assertGreaterEqual(result.ttft_seconds, SyntheticHandler.first_delay * 0.8)
        self.assertGreaterEqual(result.active_stream_seconds, SyntheticHandler.chunk_delay * 1.5)
        self.assertGreaterEqual(result.end_to_end_seconds,
                                result.ttft_seconds + result.active_stream_seconds)
        self.assertEqual(result.text, "".join(SyntheticHandler.chunks))
        self.assertEqual(result.completion_tokens, 16)
        self.assertEqual(result.tokens_per_target_pass, 4)
        self.assertEqual(result.content_events, 3)
        self.assertEqual(result.finish_reason, "stop")
        request = SyntheticHandler.requests_seen[0]
        self.assertEqual(request["temperature"], 0)
        self.assertFalse(request["chat_template_kwargs"]["enable_thinking"])
        self.assertTrue(request["stream_options"]["include_usage"])

    def test_missing_usage_does_not_guess_token_counts(self) -> None:
        SyntheticHandler.include_usage = False
        result = measure(self.endpoint, "fixture")
        self.assertIsNone(result.completion_tokens)
        self.assertIsNone(result.approximate_decode_tokens_per_second)
        self.assertIsNone(result.end_to_end_tokens_per_second)

    def test_warmup_excluded_and_requests_fixed(self) -> None:
        report = benchmark(self.endpoint, "fixture", warmups=2, runs=2,
                           max_tokens=64, mode="synthetic")
        self.assertEqual(len(SyntheticHandler.requests_seen), 4)
        self.assertEqual(len(report["measurements"]), 2)
        self.assertEqual(report["discarded_warmups"], 2)
        self.assertTrue(all(r["max_tokens"] == 64 for r in SyntheticHandler.requests_seen))
        self.assertEqual(report["mode"], "synthetic")
        json.dumps(report, allow_nan=False)

    def test_fixture_header_overrides_live_label(self) -> None:
        report = benchmark(self.endpoint, "fixture", warmups=0, runs=1)
        self.assertEqual(report["mode"], "synthetic")
        self.assertEqual(report["measurements"][0]["mode"], "synthetic")

    def test_multiline_sse_comments_done(self) -> None:
        records = BytesIO(b': heartbeat\r\nevent: message\r\ndata: {"choices":\r\n'
                          b'data: []}\r\n\r\ndata: [DONE]\r\n\r\n'
                          b'data: {"ignored": true}\r\n\r\n')
        self.assertEqual(list(sse_events(records)), [{"choices": []}])

    def test_reject_credentials_in_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "credentials"):
            measure("http://user:secret@127.0.0.1:1/v1/chat/completions", "fixture")


if __name__ == "__main__":
    unittest.main()
