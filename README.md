# trie

`trie` stands for trace replay inference evaluation and is a lightweight benchmarking harness that exercises OpenAI-compatible inference servers with synthetic workloads derived from production traces. It targets backends like vLLM, SGLang, and TensorRT-LLM.

Most inference benchmarks test prefill-heavy or decode-heavy workloads (e.g. 1k/8k, 1k/1k, 8k/1k). However, real agentic workloads look very different: they're multi-turn, have high per-turn prefill from tool outputs, and put increasing pressure on KV cache management as context grows.

## Quick start

From the CLI:

```bash
uv run trie \
  workload_path=workload.jsonl \
  endpoint=http://localhost:8000/v1 \
  model=deepseek-ai/DeepSeek-R1 \
  concurrency=8 \
  duration=300 \
  stream=True \
  num_gpus=8
```

CLI arguments use the `RunArgs` field names directly, so multiword arguments
should be passed with underscores such as `tokenizer_model=...`.

`model` is the name sent to the inference endpoint. `tokenizer_model` is
loaded separately via `transformers.AutoTokenizer.from_pretrained(...)` to
generate synthetic prompts with the requested token lengths. If `model` is
not a valid Hugging Face model ID or local checkpoint/tokenizer path, pass
`tokenizer_model=...` explicitly.

From Python:

```python
from trie import Client

client = Client(
    endpoint="http://localhost:8000/v1",
    model="deepseek-ai/DeepSeek-R1",
)
client.sync_run("workload.jsonl", concurrency=8, duration=300, stream=True)
# run() is async if you're already in an event loop
```

`duration` is the deadline for launching new traces. Once the benchmark
reaches that limit, the client stops admitting new work and cancels all
in-flight traces immediately.

Before starting a benchmark, make sure the engine is idle and not serving
leftover traffic from earlier runs. Starting from a non-idle state can skew
cache behavior, warmup, and steady-state throughput measurements.

## Backend requirements

- `extra_body={"ignore_eos": True}`: the harness uses this to force fixed-length
  generations (effectively `min_tokens = max_tokens`). Backends must support
  this extension.
- Cache-hit metrics require the server to return
  `usage.prompt_tokens_details.cached_tokens` on completion responses:
  - **SGLang**: launch with `--enable-cache-report`.
  - **vLLM**: launch with `--enable-prompt-tokens-details`.
- The client-side `transformers` tokenizer should match what the server uses.
  Mismatches can miscount synthetic prompts and cause context-length errors.

## Workload format

Each JSONL row defines one trace:

- `num_turns` — number of tool-use turns
- `input_prompt_length` — initial user prompt token length
- `assistant_response_length` — per-turn assistant tokens (list of length `num_turns`)
- `tool_call_output_length` — per-turn tool result tokens (list of length `num_turns`)
- `tool_call_latency` — per-turn simulated delay in seconds (list of length `num_turns`)
- `final_assistant_response_length` — final assistant response token length after all tool-use turns

Example:

```json
{"num_turns": 2, "input_prompt_length": 32, "assistant_response_length": [16, 20], "tool_call_output_length": [8, 12], "tool_call_latency": [0.0, 0.0], "final_assistant_response_length": 64}
```

A trace produces `num_turns + 1` completion requests: one per tool-use turn plus a final turn after the last tool result.

## Example output

```
[info     ] starting benchmark             concurrency=24 duration=300.0 model=/data/models/DeepSeek-R1 num_gpus=8 workload_templates=8192
[warning  ] benchmark interrupted
[info     ] benchmark complete             completed_requests=8 failed_requests=0 wall_time_s=109.25 ...

                                                 Per-trace metrics
┏━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        ┃             ┃          ┃           ┃                    ┃                    ┃ Eligible cache hit rate ┃
┃ Metric ┃ Latency (s) ┃ TTFT (s) ┃ TTFAT (s) ┃ Decode TPS (tok/s) ┃ Cache hit rate (%) ┃                     (%) ┃
┡━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ mean   │      74.572 │    2.231 │    28.510 │              23.69 │               53.0 │                    96.6 │
│ min    │      37.151 │    2.105 │     4.666 │              19.27 │               23.3 │                    93.7 │
│ p50    │      77.915 │    2.181 │    16.585 │              20.71 │               43.9 │                    96.9 │
│ p90    │      99.497 │    2.450 │    63.775 │              30.72 │               85.7 │                    98.6 │
│ p95    │     103.044 │    2.452 │    66.554 │              32.30 │               85.8 │                    98.7 │
│ p99    │     105.882 │    2.454 │    68.778 │              33.57 │               85.9 │                    98.8 │
│ max    │     106.592 │    2.455 │    69.334 │              33.89 │               86.0 │                    98.8 │
└────────┴─────────────┴──────────┴───────────┴────────────────────┴────────────────────┴─────────────────────────┘

                                     Workload metrics
                                completed=8/8  trace/s=0.07
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                ┃  Overall ┃ Last 30s Window ┃ Steady State ┃ Steady State / GPU ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ total prompt tok/s    │ 24952.61 │        30315.66 │     30263.17 │            3782.90 │
│ cached prompt tok/s   │ 19720.40 │        25550.18 │     24628.21 │            3078.53 │
│ uncached prompt tok/s │  5232.21 │         4765.47 │      5634.96 │             704.37 │
│ completion tok/s      │   278.97 │          280.88 │       281.12 │              35.14 │
└───────────────────────┴──────────┴─────────────────┴──────────────┴────────────────────┘
```

## Metrics

### Per-trace

- `Latency (s)` — end-to-end latency from the first request of a trace to the final response.
- `TTFT (s)` — (streaming) time to the first streamed token of the first request.
- `TTFAT (s)` — (streaming) time from trace start to the first streamed token of the *final* request. The user-visible first token in an agent that hides intermediate tool turns.
- `Decode TPOT (ms/tok)` — (streaming) inverse of mean post-TTFT decode throughput across the trace's requests. Higher percentile rows are the slow tail, so p95/p99 represent worse decode cases.
- `ITL (ms)` — (streaming) client-observed inter-token latency, reported across observed token intervals.
- `Cache hit rate (%)` — server-reported `cached_prompt_tokens / prompt_tokens` over all requests in a trace.
- `Eligible cache hit rate (%)` — same numerator, but the denominator is restricted to prompt tokens expected to be cacheable. This excludes the initial prompt and, on each turn, the tool output newly appended on that request.
  Formula: `sum_i cached_prompt_tokens_i / sum_i eligible_prompt_tokens_i`,
  where `eligible_prompt_tokens_0 = 0` and `eligible_prompt_tokens_i = prompt_tokens_{i-1} + completion_tokens_{i-1}` for `i > 0`.

### Workload

- `trace/s` — completed traces per wall-clock second.
- `total prompt tok/s`, `cached prompt tok/s`, `uncached prompt tok/s` — aggregate prompt-token throughput, split by what the synthetic workload accounting expects to be cached vs. new.
- `completion tok/s` — aggregate completion-token throughput.

Each is reported under four columns:

- `Overall` — totals over the full benchmark wall time.
- `Last 30s Window` — slope of cumulative token counts over the most recent 30 seconds.
- `Steady State` — **the headline throughput metric**. Slope after dropping the first 20% of wall time as warmup. Avoids dilution from ramp-up and drain when fewer than `concurrency` traces are in flight. With `concurrency > 1` the completion curve depends on finish order, so the metric has small run-to-run variance even at fixed seed.
- `Steady State / GPU` — `Steady State / num_gpus` when `num_gpus` is provided.

Prompt-token throughputs use the synthetic workload accounting; cache-hit metrics use server-reported usage. Divergence implies a tokenizer mismatch between client and server.

## Known limitations

- Synthetic prompts are freshly random per trace, so cross-trace prefix sharing (e.g. a common system prompt or tool definition block) is not modeled and cache hit rates can be lower than in a deployment where such prefixes are shared.
- `Decode TPOT (ms/tok)` and `ITL (ms)` are client-observed streaming metrics. If a backend buffers multiple tokens into one streamed chunk, `ITL (ms)` spreads the elapsed time since the prior chunk evenly across newly observed tokens in that chunk, and intervals inside the first streamed chunk are not observable.
- The pinned `transformers==4.57.6` is required, at least for DeepSeek-R1, to match the tokenizer used by vLLM and SGLang as of April 2026. Other `transformers` versions can produce subtly different token counts and cause prompt-accounting drift.
