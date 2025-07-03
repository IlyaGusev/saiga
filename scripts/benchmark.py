#!/usr/bin/env python
"""benchmark_vllm.py

Benchmark chat completions served by a vLLM OpenAI‑compatible endpoint.

Example usage:
    python benchmark_vllm.py run \
        --dataset ./data/benchmark.jsonl \
        --concurrency 8 \
        --model "llama-3-8b-instruct" \
        --max_tokens 512
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from statistics import mean, median

import fire
import httpx
from tqdm import tqdm


async def _request_chat_completion(
    client: httpx.AsyncClient, url: str, payload: dict
) -> dict:
    t0 = time.perf_counter()
    response = await client.post(url, json=payload)
    latency = time.perf_counter() - t0
    response.raise_for_status()

    data = response.json()
    usage = data.get("usage", {})
    return {
        "response": response,
        "latency": latency,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


async def _bounded_worker(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
):
    async with semaphore:
        return await _request_chat_completion(client, url, payload)


async def _run_async(
    conversations: list[list[dict]],
    concurrency: int,
    base_url: str,
    model: str,
    max_tokens: int,
) -> list[dict]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async with httpx.AsyncClient(timeout=None) as client:
        tasks = [
            asyncio.create_task(
                _bounded_worker(
                    sem,
                    client,
                    url,
                    {
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.6,
                        "top_p": 0.9,
                        "stream": False,
                    },
                )
            )
            for messages in conversations
        ]

        for coro in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Benchmarking"
        ):
            results.append(await coro)

    return results


def run(
    dataset: str,
    model: str,
    concurrency: int = 1,
    max_tokens: int = 1024,
    base_url: str = "http://localhost:8000/v1",
    nrows: int = 100,
):
    """Run a chat‑completion benchmark against a vLLM server.

    Args:
        dataset: Path to a JSONL file where each line is ``{"messages": [...]} ``.
        concurrency: Number of concurrent in‑flight requests.
        model: Model name passed to the endpoint (informational for vLLM).
        max_tokens: ``max_tokens`` parameter per request.
        base_url: Base URL of the OpenAI‑compatible API (default: localhost vLLM).
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    ds_path = Path(dataset)
    if not ds_path.is_file():
        raise FileNotFoundError(dataset)

    # Parse dataset
    conversations: list[list[dict]] = []
    with ds_path.open("r", encoding="utf‑8") as fp:
        for lineno, line in enumerate(fp, 1):
            if not line.strip():
                continue  # skip blank lines
            try:
                obj = json.loads(line)
                messages = obj["messages"]
                if messages[-1]["role"] == "assistant":
                    messages = messages[:-1]
                conversations.append(messages)
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid JSONL on line {lineno}: {exc}") from exc

    if not conversations:
        raise ValueError("Dataset is empty – no messages found.")

    conversations = conversations[:nrows]
    print(f"Loaded {len(conversations)} conversations from {ds_path}.")
    print(
        f"Concurrency: {concurrency} | Model: {model} | Max tokens: {max_tokens}\n"
    )

    # Run benchmark
    t_start = time.perf_counter()
    stats = asyncio.run(
        _run_async(
            conversations,
            concurrency=concurrency,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
        )
    )
    wall_time = time.perf_counter() - t_start

    # Aggregate metrics
    latencies = [s["latency"] for s in stats]
    avg_lat, med_lat = mean(latencies), median(latencies)
    p95_lat = sorted(latencies)[int(0.95 * len(latencies)) - 1]

    prompt_tokens = sum(s["prompt_tokens"] for s in stats)
    completion_tokens = sum(s["completion_tokens"] for s in stats)
    total_tokens = sum(s["total_tokens"] for s in stats)

    rps = len(stats) / wall_time
    completion_tps = completion_tokens / wall_time if wall_time else 0

    example_response = stats[-1]["response"]

    # Pretty print results
    print("=== Results ===")
    print("Example response:", example_response.json())
    print(f"Requests             : {len(stats)}")
    print(f"Total wall time      : {wall_time:.2f} s")
    print(f"Throughput           : {rps:.2f} req/s")
    print("Latency (s)")
    print(f"  avg | p50 | p95     : {avg_lat:.3f} | {med_lat:.3f} | {p95_lat:.3f}")
    print(f"Prompt tokens        : {prompt_tokens}")
    print(f"Completion tokens    : {completion_tokens}")
    print(f"Total tokens         : {total_tokens}")
    print(f"Completion token throughput     : {completion_tps:.2f} tok/s")


if __name__ == "__main__":
    fire.Fire(run)

