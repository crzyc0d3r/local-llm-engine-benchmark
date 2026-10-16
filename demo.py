"""Run the complete measurement path against a synthetic loopback stream."""

import json
from threading import Thread

from benchmark import benchmark
from synthetic_server import create_server


def main() -> None:
    with create_server() as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            report = benchmark(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                "synthetic-fixture", engine="synthetic", mode="synthetic",
                warmups=1, runs=2,
            )
            print("SYNTHETIC STREAM: results validate the harness, not an LLM or hardware.")
            print(json.dumps(report, indent=2))
        finally:
            server.shutdown()
            worker.join()


if __name__ == "__main__":
    main()
