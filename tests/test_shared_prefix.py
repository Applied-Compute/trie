"""Shared-prefix workloads, exercised against an in-process fake completions
server with a prefix cache. No model or network is needed: the tokenizer is a
stub whose tokens are space-separated words, so token counts are exact."""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from openai import AsyncOpenAI

import trie.client as client_module
from trie import Client
from trie.results import BenchmarkResult, ServerMetrics
from trie.types import Turn, Workload

ROOT = Path(__file__).resolve().parents[1]


class WordTokenizer:
    """Stand-in for TokenizerManager: every token is one ' t<id>' word."""

    def __init__(self, model_name: str, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def get_prompt(self, prompt_length: int, *, max_rounds: int = 10) -> str:
        ids = self._rng.integers(1_000_000, size=prompt_length)
        return "".join(f" t{i}" for i in ids)

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class PrefixCachingServer:
    """Fake /v1/completions endpoint. Like vLLM/SGLang it prepends a BOS
    token, reuses the longest prefix of any earlier request (prompt plus
    completion), always recomputes the last prompt token, and reports the
    reuse as usage.prompt_tokens_details.cached_tokens."""

    def __init__(self) -> None:
        self.seen: set[tuple[str, ...]] = set()
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        tokens = ("<bos>", *body["prompt"].split())
        cached = 0
        # A match of BOS alone is not reuse, so the shortest hit is 2 tokens.
        for n in range(len(tokens) - 1, 1, -1):
            if tokens[:n] in self.seen:
                cached = n
                break
        completion = tuple(f"c{i}" for i in range(body["max_tokens"]))
        full = tokens + completion
        for n in range(1, len(full) + 1):
            self.seen.add(full[:n])
        return httpx.Response(
            200,
            json={
                "id": "cmpl",
                "object": "text_completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "text": "".join(f" {t}" for t in completion),
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(tokens),
                    "completion_tokens": len(completion),
                    "total_tokens": len(tokens) + len(completion),
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
            },
        )


@pytest.fixture
def server() -> PrefixCachingServer:
    return PrefixCachingServer()


@pytest.fixture
def client(monkeypatch, server: PrefixCachingServer) -> Client:
    monkeypatch.setattr(client_module, "TokenizerManager", WordTokenizer)
    c = Client(endpoint="http://fake/v1", model="m", api_key="x", seed=0)
    c._client = AsyncOpenAI(
        base_url="http://fake/v1",
        api_key="x",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(server.handler)),
    )
    return c


def workload(prefix: int = 0, prefix_id: str = "default", turns: int = 2) -> Workload:
    return Workload(
        input_prompt_length=600,
        turns=[Turn(20, 40, 0.0) for _ in range(turns)],
        final_assistant_response_length=10,
        shared_prefix_length=prefix,
        shared_prefix_id=prefix_id,
    )


async def run_sequentially(
    client: Client, workloads: list[Workload]
) -> BenchmarkResult:
    result = BenchmarkResult()
    start = time.perf_counter()
    for w in workloads:
        await client._run_workload(
            w, benchmark_start=start, result=result, stream=False, refresh=lambda: None
        )
    return result


# --- workload format -------------------------------------------------------


def test_jsonl_fields_are_optional_and_backward_compatible():
    row = {
        "num_turns": 1,
        "input_prompt_length": 32,
        "assistant_response_length": [16],
        "tool_call_output_length": [8],
        "tool_call_latency": [0.0],
        "final_assistant_response_length": 64,
    }
    w = Workload.from_jsonl_row(row)
    assert (w.shared_prefix_length, w.shared_prefix_id) == (0, "default")
    w = Workload.from_jsonl_row(
        row | {"shared_prefix_length": 20, "shared_prefix_id": 7}
    )
    assert (w.shared_prefix_length, w.shared_prefix_id) == (20, "7")


@pytest.mark.parametrize("bad", [-1, 33])
def test_shared_prefix_must_fit_in_input_prompt(bad):
    with pytest.raises(ValueError, match="shared_prefix_length"):
        Workload(32, [], 8, shared_prefix_length=bad)


# --- prompt construction -----------------------------------------------------


def test_traces_in_a_group_share_the_exact_prefix_and_nothing_more(client):
    a = client._initial_prompt(workload(400, "x")).split()
    b = client._initial_prompt(workload(400, "x")).split()
    other = client._initial_prompt(workload(400, "y")).split()
    # 599 synthesized tokens + the server's BOS = input_prompt_length.
    assert len(a) == len(b) == len(other) == 599
    assert a[:399] == b[:399]  # 400-token prefix including BOS
    assert a[399:] != b[399:]
    assert a[:10] != other[:10]


def test_without_a_shared_prefix_traces_share_nothing(client):
    a = client._initial_prompt(workload()).split()
    b = client._initial_prompt(workload()).split()
    assert len(a) == 599 and a[0] != b[0]


# --- cache accounting against a prefix-caching server --------------------------


def test_first_trace_is_cold_later_traces_hit_the_shared_prefix(client, server):
    result = asyncio.run(run_sequentially(client, [workload(400)] * 3))
    first, second, third = result.server_metrics
    first_requests = server.requests[0::3]
    assert len(first_requests) == 3

    assert first.first_request_eligible_prompt_tokens == 0
    assert second.first_request_eligible_prompt_tokens == 400
    # The server reused the whole shared prefix (incl. BOS) on traces 2 and 3,
    # and the eligible denominator counts exactly that, so the rate is 100%.
    for m in (second, third):
        assert m.eligible_cache_hit_rate == pytest.approx(1.0)
    # Trace 1 had nothing to reuse on its first request.
    assert first.eligible_cache_hit_rate == pytest.approx(1.0)
    assert first.cached_tokens < second.cached_tokens
    assert second.cached_tokens - first.cached_tokens == 400


def test_baseline_without_shared_prefix_has_no_first_request_hits(client):
    result = asyncio.run(run_sequentially(client, [workload()] * 3))
    hits = [m.cached_tokens for m in result.server_metrics]
    assert hits[0] == hits[1] == hits[2]
    assert all(
        m.first_request_eligible_prompt_tokens == 0 for m in result.server_metrics
    )


def test_separate_groups_do_not_warm_each_other(client):
    result = asyncio.run(
        run_sequentially(
            client, [workload(400, "a"), workload(400, "b"), workload(400, "a")]
        )
    )
    _, b1, a2 = result.server_metrics
    assert b1.first_request_eligible_prompt_tokens == 0
    assert a2.first_request_eligible_prompt_tokens == 400


def test_single_request_traces_are_warmed_by_their_final_request(client):
    result = asyncio.run(run_sequentially(client, [workload(400, turns=0)] * 2))
    cold, warm = result.server_metrics
    assert cold.cached_tokens == 0
    assert warm.cached_tokens == 400
    assert warm.eligible_cache_hit_rate == pytest.approx(1.0)


def test_synthetic_throughput_accounting_counts_warm_prefix_as_cached(client):
    result = asyncio.run(run_sequentially(client, [workload(400)] * 2))
    w = workload(400)
    total_prompt = 2 * sum(
        w.cumulative_prompt_tokens(i) for i in range(len(w.turns) + 1)
    )
    assert result.completed_prompt_tokens == total_prompt
    baseline = asyncio.run(run_sequentially(client, [workload()] * 2))
    assert result.cached_prompt_tokens - baseline.cached_prompt_tokens == 400


def test_eligible_tokens_start_from_first_request_value():
    m = ServerMetrics(first_request_eligible_prompt_tokens=50)

    class U:
        prompt_tokens = 100
        completion_tokens = 10
        prompt_tokens_details = None

    m.record_usage(U())
    m.record_usage(U())
    assert m.eligible_prompt_tokens == 50 + 110


def test_each_run_starts_cold(client, server):
    asyncio.run(client.run([workload(400)], concurrency=1, duration=0.3))
    assert client._warm_prefixes == {("default", 400)}
    first_prefix = client._shared_prefixes[("default", 400)]
    asyncio.run(client.run([workload(400)], concurrency=1, duration=0.3))
    assert client._shared_prefixes[("default", 400)] != first_prefix


def test_full_run_reaches_full_eligible_hit_rate(client):
    result = asyncio.run(client.run([workload(400)], concurrency=4, duration=0.5))
    rates = [m.eligible_cache_hit_rate for m in result.server_metrics]
    assert len(rates) > 4
    assert min(rates) == pytest.approx(1.0)


# --- generator script --------------------------------------------------------


def test_generate_workload_assigns_prefix_groups_round_robin(tmp_path):
    out = tmp_path / "w.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_workload.py"),
            "num_traces=5",
            f"out={out}",
            "input_prompt_length=1000",
            "shared_prefix_length=600",
            "num_shared_prefixes=2",
        ],
        check=True,
        capture_output=True,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["shared_prefix_id"] for r in rows] == [
        "prefix-0",
        "prefix-1",
        "prefix-0",
        "prefix-1",
        "prefix-0",
    ]
    assert all(Workload.from_jsonl_row(r).shared_prefix_length == 600 for r in rows)
