"""Concurrent HTTP benchmark for a running MiniVLLM online server."""

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def post_completion(url: str, index: int, prompt: str, max_tokens: int) -> dict:
    payload = json.dumps(
        {
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": max_tokens,
            "request_id": f"benchmark-{index}",
        }
    ).encode()
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read())
    return {
        "latency_seconds": time.perf_counter() - started,
        "completion_tokens": result["usage"]["completion_tokens"],
    }


def get_json(url: str, path: str) -> dict:
    with urllib.request.urlopen(f"{url}{path}") as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt",
        default="Explain why continuous batching improves inference throughput.",
    )
    args = parser.parse_args()

    wall_started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                post_completion,
                args.url,
                index,
                args.prompt,
                args.max_tokens,
            )
            for index in range(args.requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    wall_seconds = time.perf_counter() - wall_started
    stats = get_json(args.url, "/stats")
    active_batch_sizes = [
        step["active_batch_size"] for step in stats.get("recent_decode_steps", [])
    ]
    total_tokens = sum(result["completion_tokens"] for result in results)
    latencies = [result["latency_seconds"] for result in results]
    print(
        json.dumps(
            {
                "requests": len(results),
                "concurrency": args.concurrency,
                "wall_seconds": wall_seconds,
                "requests_per_second": len(results) / wall_seconds,
                "completion_tokens_per_second": total_tokens / wall_seconds,
                "mean_latency_seconds": statistics.mean(latencies),
                "p95_latency_seconds": sorted(latencies)[
                    min(len(latencies) - 1, int(len(latencies) * 0.95))
                ],
                "max_observed_decode_batch_size": max(active_batch_sizes, default=0),
                "observed_continuous_batching": any(
                    size > 1 for size in active_batch_sizes
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
