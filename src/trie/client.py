import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import cycle

import structlog
from openai import AsyncOpenAI
from rich.live import Live

from trie.reporting import build_progress_lines, console, log_summary
from trie.results import (
    BenchmarkResult,
    ServerMetrics,
    StreamAccumulator,
)
from trie.tokenizer_manager import TokenizerManager
from trie.types import Workload

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RequestResult:
    text: str
    output_tokens: int


def load_workloads(workload: list[Workload] | str) -> list[Workload]:
    if not isinstance(workload, str):
        return workload
    with open(workload, "r", encoding="utf-8") as f:
        return [Workload.from_jsonl_row(json.loads(line)) for line in f]


class Client:
    """Runs synthetic multi-turn workloads against an OpenAI-compatible endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        tokenizer_model: str | None = None,
        api_key: str | None = None,
        seed: int | None = None,
        max_retries: int = 2,
        timeout: float = 600.0,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
        )
        self._tokenizer_manager = TokenizerManager(tokenizer_model or model, seed=seed)
        self._result: BenchmarkResult | None = None
        self._benchmark_start: float | None = None
        self._stopping_run = False

    async def _execute_stream_request(
        self,
        prompt: str,
        max_tokens: int,
        *,
        request_id: str,
        trace_start: float,
        metrics: ServerMetrics,
        stream_acc: StreamAccumulator,
        logprobs: int | None,
    ) -> RequestResult:
        request_start = time.perf_counter()
        request_kwargs = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": {"ignore_eos": True},
            "extra_headers": {"X-SMG-Routing-Key": request_id},
        }
        if logprobs is not None:
            request_kwargs["logprobs"] = logprobs
        stream = await self._client.completions.create(**request_kwargs)
        text_parts: list[str] = []
        last_token_at_s: float | None = None
        inter_token_latencies_ms: list[float] = []
        ttft_s: float | None = None
        first_token_offset_s: float | None = None
        usage = None
        async for chunk in stream:
            if chunk.choices:
                text = chunk.choices[0].text or ""
                if text:
                    chunk_at_s = time.perf_counter()
                    if ttft_s is None:
                        ttft_s = chunk_at_s - request_start
                        first_token_offset_s = chunk_at_s - trace_start
                    text_parts.append(text)
                    token_count = self._tokenizer_manager.count_tokens(text)
                    if token_count > 0:
                        if last_token_at_s is not None:
                            gap_ms = (
                                (chunk_at_s - last_token_at_s) * 1000.0 / token_count
                            )
                            inter_token_latencies_ms.extend([gap_ms] * token_count)
                        last_token_at_s = chunk_at_s
            if chunk.usage is not None:
                usage = chunk.usage
        latency_s = time.perf_counter() - request_start
        if ttft_s is None or first_token_offset_s is None:
            raise ValueError("stream produced no chunks with choices")
        if usage is not None:
            metrics.record_usage(usage)
        output_tokens = usage.completion_tokens if usage is not None else max_tokens
        stream_acc.record_turn_stream_metrics(
            ttft_s,
            latency_s,
            output_tokens,
            first_token_offset_s=first_token_offset_s,
            inter_token_latencies_ms=inter_token_latencies_ms,
        )
        return RequestResult("".join(text_parts), output_tokens)

    async def _execute_request(
        self,
        prompt: str,
        max_tokens: int,
        *,
        request_id: str,
        stream: bool,
        trace_start: float,
        metrics: ServerMetrics,
        stream_acc: StreamAccumulator,
        logprobs: int | None,
    ) -> RequestResult:
        if stream:
            return await self._execute_stream_request(
                prompt,
                max_tokens,
                request_id=request_id,
                trace_start=trace_start,
                metrics=metrics,
                stream_acc=stream_acc,
                logprobs=logprobs,
            )

        request_kwargs = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "extra_body": {"ignore_eos": True},
            "extra_headers": {"X-SMG-Routing-Key": request_id},
        }
        if logprobs is not None:
            request_kwargs["logprobs"] = logprobs
        response = await self._client.completions.create(**request_kwargs)
        if response.usage is None:
            raise ValueError("response.usage must not be None")
        metrics.record_usage(response.usage)
        return RequestResult(
            response.choices[0].text or "",
            response.usage.completion_tokens,
        )

    async def _run_workload(
        self,
        workload: Workload,
        *,
        benchmark_start: float,
        result: BenchmarkResult,
        request_id: str,
        stream: bool,
        logprobs: int | None,
        refresh: Callable[[], None],
    ) -> None:
        request_start = time.perf_counter()
        turns_completed = 0
        metrics = ServerMetrics()
        stream_acc = StreamAccumulator()
        output_tokens_per_turn: list[int] = []
        try:
            # Generate one fewer token than requested to compensate for BOS token
            # which is added by vLLM, SGLang, and TensorRT-LLM.
            prompt = self._tokenizer_manager.get_prompt(
                max(workload.input_prompt_length - 1, 0)
            )
            for turn_index, turn in enumerate(workload.turns):
                response = await self._execute_request(
                    prompt,
                    turn.assistant_response_length,
                    request_id=request_id,
                    stream=stream,
                    trace_start=request_start,
                    metrics=metrics,
                    stream_acc=stream_acc,
                    logprobs=logprobs,
                )
                output_tokens_per_turn.append(response.output_tokens)
                result.record_turn(
                    workload,
                    turn_index,
                    timestamp=time.perf_counter() - benchmark_start,
                )
                turns_completed = turn_index + 1
                refresh()
                prompt += response.text
                await asyncio.sleep(turn.tool_call_latency)
                prompt += self._tokenizer_manager.get_prompt(
                    turn.tool_call_output_length
                )

            response = await self._execute_request(
                prompt,
                workload.final_assistant_response_length,
                request_id=request_id,
                stream=stream,
                trace_start=request_start,
                metrics=metrics,
                stream_acc=stream_acc,
                logprobs=logprobs,
            )
            output_tokens_per_turn.append(response.output_tokens)
            if not stream or metrics.prompt_tokens > 0:
                result.server_metrics.append(metrics)
            else:
                logger.warning("stream response usage missing; omitting server metrics")
            event_timestamp = time.perf_counter()
            result.record_success(
                workload,
                latency=event_timestamp - request_start,
                timestamp=event_timestamp - benchmark_start,
                request_id=request_id,
                output_tokens_per_turn=output_tokens_per_turn,
            )
            if stream:
                stream_acc.commit(result)
        except asyncio.CancelledError:
            return
        except Exception as e:
            if self._stopping_run:
                return
            result.record_failure()
            logger.warning(
                "request failed",
                error=str(e),
                request_id=request_id,
                turn=turns_completed,
            )
        refresh()

    async def run(
        self,
        workload: list[Workload] | str,
        concurrency: int | None = None,
        duration: float = 300.0,
        arrival_rate: float | None = None,
        duration_update_interval: float | None = None,
        num_gpus: int | None = None,
        stream: bool = False,
        logprobs: int | None = None,
        output_metrics_path: str | None = None,
    ) -> BenchmarkResult:
        workload_list = load_workloads(workload)
        if (concurrency is None) == (arrival_rate is None):
            raise ValueError("specify exactly one of concurrency or arrival_rate")
        if concurrency is not None and concurrency <= 0:
            raise ValueError("concurrency must be greater than 0")
        if arrival_rate is not None and arrival_rate <= 0:
            raise ValueError("arrival_rate must be greater than 0")
        if duration_update_interval is not None and duration_update_interval <= 0:
            raise ValueError("duration_update_interval must be greater than 0")
        benchmark_log_fields = {
            "model": self._model,
            "workload_templates": len(workload_list),
            "duration": duration,
            "num_gpus": num_gpus,
        }
        if concurrency is not None:
            benchmark_log_fields["concurrency"] = concurrency
        if arrival_rate is not None:
            benchmark_log_fields["arrival_rate"] = arrival_rate
        if duration_update_interval is not None:
            benchmark_log_fields["duration_update_interval"] = duration_update_interval
        if logprobs is not None:
            benchmark_log_fields["logprobs"] = logprobs
        logger.info(
            "starting benchmark",
            **benchmark_log_fields,
        )
        result = BenchmarkResult()
        benchmark_start = time.perf_counter()
        self._result = result
        self._benchmark_start = benchmark_start

        with Live(
            build_progress_lines(result, 0.0, duration),
            console=console,
            auto_refresh=False,
            transient=True,
        ) as live:
            next_duration_update = duration_update_interval

            def refresh() -> None:
                live.update(
                    build_progress_lines(
                        result,
                        time.perf_counter() - benchmark_start,
                        duration,
                    ),
                    refresh=True,
                )

            def log_duration_update() -> None:
                nonlocal next_duration_update
                if next_duration_update is None:
                    return
                elapsed = time.perf_counter() - benchmark_start
                if elapsed < next_duration_update:
                    return
                logger.info(
                    "benchmark in progress",
                    remaining_duration=round(max(duration - elapsed, 0.0), 1),
                )
                next_duration_update = elapsed + duration_update_interval

            active: set[asyncio.Task[None]] = set()
            workload_iter = cycle(workload_list)
            next_arrival = benchmark_start
            arrival_interval = 1.0 / arrival_rate if arrival_rate is not None else None
            next_trace_id = 0

            while time.perf_counter() - benchmark_start < duration:
                log_duration_update()
                if arrival_interval is not None:
                    done = {task for task in active if task.done()}
                    active -= done
                    for task in done:
                        task.result()

                if concurrency is not None and len(active) == concurrency:
                    timeout = None
                    if next_duration_update is not None:
                        elapsed = time.perf_counter() - benchmark_start
                        timeout = max(
                            min(next_duration_update - elapsed, duration - elapsed),
                            0.0,
                        )
                    done, active = await asyncio.wait(
                        active,
                        timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        task.result()
                    if not done:
                        continue

                if arrival_interval is not None:
                    wait_until = min(next_arrival, benchmark_start + duration)
                    if next_duration_update is not None:
                        wait_until = min(
                            wait_until,
                            benchmark_start + next_duration_update,
                        )
                    timeout = wait_until - time.perf_counter()
                    if timeout > 0:
                        await asyncio.sleep(timeout)
                        continue

                request_id = str(next_trace_id)
                next_trace_id += 1
                active.add(
                    asyncio.create_task(
                        self._run_workload(
                            next(workload_iter),
                            benchmark_start=benchmark_start,
                            result=result,
                            request_id=request_id,
                            stream=stream,
                            logprobs=logprobs,
                            refresh=refresh,
                        )
                    )
                )
                if arrival_interval is not None:
                    next_arrival += arrival_interval
            for task in active:
                task.cancel()
            if active:
                self._stopping_run = True
                try:
                    await self._client.close()
                except Exception as e:
                    logger.warning("failed to close OpenAI client", error=str(e))
                _, pending = await asyncio.wait(active, timeout=30.0)
                if pending:
                    logger.warning(
                        "timed out waiting for cancelled traces",
                        pending_traces=len(pending),
                    )
        result.wall_time = time.perf_counter() - benchmark_start
        log_summary(result, num_gpus=num_gpus)
        if output_metrics_path is not None:
            result.write_output_token_records(output_metrics_path)
            logger.info(
                "wrote output token metrics",
                output_metrics_path=output_metrics_path,
                traces=len(result.trace_output_token_records),
            )
        return result

    def sync_run(
        self,
        workload: list[Workload] | str,
        concurrency: int | None = None,
        duration: float = 300.0,
        arrival_rate: float | None = None,
        duration_update_interval: float | None = None,
        num_gpus: int | None = None,
        stream: bool = False,
        logprobs: int | None = None,
        output_metrics_path: str | None = None,
    ) -> BenchmarkResult:
        try:
            return asyncio.run(
                self.run(
                    workload,
                    concurrency,
                    duration,
                    arrival_rate=arrival_rate,
                    duration_update_interval=duration_update_interval,
                    num_gpus=num_gpus,
                    stream=stream,
                    logprobs=logprobs,
                    output_metrics_path=output_metrics_path,
                )
            )
        except KeyboardInterrupt:
            logger.warning("benchmark interrupted")
            assert self._result is not None and self._benchmark_start is not None
            result = self._result
            result.wall_time = time.perf_counter() - self._benchmark_start
            log_summary(result, num_gpus=num_gpus)
            if output_metrics_path is not None:
                result.write_output_token_records(output_metrics_path)
                logger.info(
                    "wrote output token metrics",
                    output_metrics_path=output_metrics_path,
                    traces=len(result.trace_output_token_records),
                )
            return result
        finally:
            self._stopping_run = False
            self._result = None
            self._benchmark_start = None
