import asyncio
import json
import time
from collections.abc import Callable
from itertools import cycle

import structlog
from openai import AsyncOpenAI, OpenAIError
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

    async def _execute_stream_request(
        self,
        prompt: str,
        max_tokens: int,
        *,
        trace_start: float,
        metrics: ServerMetrics,
        stream_acc: StreamAccumulator,
    ) -> str:
        request_start = time.perf_counter()
        stream = await self._client.completions.create(
            model=self._model,
            prompt=prompt,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"ignore_eos": True},
        )
        text_parts: list[str] = []
        ttft_s: float | None = None
        first_token_offset_s: float | None = None
        usage = None
        async for chunk in stream:
            if chunk.choices:
                if ttft_s is None:
                    first_token_at_s = time.perf_counter()
                    ttft_s = first_token_at_s - request_start
                    first_token_offset_s = first_token_at_s - trace_start
                text_parts.append(chunk.choices[0].text or "")
            if chunk.usage is not None:
                usage = chunk.usage
        latency_s = time.perf_counter() - request_start
        if ttft_s is None or first_token_offset_s is None:
            raise ValueError("stream produced no chunks with choices")
        if usage is not None:
            metrics.record_usage(usage)
        stream_acc.record_turn_stream_metrics(
            ttft_s,
            latency_s,
            usage.completion_tokens if usage is not None else max_tokens,
            first_token_offset_s=first_token_offset_s,
        )
        return "".join(text_parts)

    async def _execute_request(
        self,
        prompt: str,
        max_tokens: int,
        *,
        stream: bool,
        trace_start: float,
        metrics: ServerMetrics,
        stream_acc: StreamAccumulator,
    ) -> str:
        if stream:
            return await self._execute_stream_request(
                prompt,
                max_tokens,
                trace_start=trace_start,
                metrics=metrics,
                stream_acc=stream_acc,
            )

        response = await self._client.completions.create(
            model=self._model,
            prompt=prompt,
            max_tokens=max_tokens,
            extra_body={"ignore_eos": True},
        )
        if response.usage is None:
            raise ValueError("response.usage must not be None")
        metrics.record_usage(response.usage)
        return response.choices[0].text or ""

    async def _run_workload(
        self,
        workload: Workload,
        *,
        benchmark_start: float,
        result: BenchmarkResult,
        stream: bool,
        refresh: Callable[[], None],
    ) -> None:
        request_start = time.perf_counter()
        turns_completed = 0
        metrics = ServerMetrics()
        stream_acc = StreamAccumulator()
        try:
            # Generate one fewer token than requested to compensate for BOS token
            # which is added by vLLM, SGLang, and TensorRT-LLM.
            prompt = self._tokenizer_manager.get_prompt(
                max(workload.input_prompt_length - 1, 0)
            )
            for turn_index, turn in enumerate(workload.turns):
                response_text = await self._execute_request(
                    prompt,
                    turn.assistant_response_length,
                    stream=stream,
                    trace_start=request_start,
                    metrics=metrics,
                    stream_acc=stream_acc,
                )
                result.record_turn(
                    workload,
                    turn_index,
                    timestamp=time.perf_counter() - benchmark_start,
                )
                turns_completed = turn_index + 1
                refresh()
                prompt += response_text
                await asyncio.sleep(turn.tool_call_latency)
                prompt += self._tokenizer_manager.get_prompt(
                    turn.tool_call_output_length
                )

            await self._execute_request(
                prompt,
                workload.final_assistant_response_length,
                stream=stream,
                trace_start=request_start,
                metrics=metrics,
                stream_acc=stream_acc,
            )
            if not stream or metrics.prompt_tokens > 0:
                result.server_metrics.append(metrics)
            else:
                logger.warning("stream response usage missing; omitting server metrics")
            event_timestamp = time.perf_counter()
            result.record_success(
                workload,
                latency=event_timestamp - request_start,
                timestamp=event_timestamp - benchmark_start,
            )
            if stream:
                stream_acc.commit(result)
        except asyncio.CancelledError:
            return
        except OpenAIError as e:
            result.record_failure()
            logger.warning("request failed", error=str(e), turn=turns_completed)
        refresh()

    async def run(
        self,
        workload: list[Workload] | str,
        concurrency: int,
        duration: float,
        duration_update_interval: float | None = None,
        num_gpus: int | None = None,
        stream: bool = False,
    ) -> BenchmarkResult:
        workload_list = load_workloads(workload)
        if duration_update_interval is not None and duration_update_interval <= 0:
            raise ValueError("duration_update_interval must be greater than 0")
        benchmark_log_fields = {
            "model": self._model,
            "workload_templates": len(workload_list),
            "concurrency": concurrency,
            "duration": duration,
            "num_gpus": num_gpus,
        }
        if duration_update_interval is not None:
            benchmark_log_fields["duration_update_interval"] = duration_update_interval
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
            for item in cycle(workload_list):
                if time.perf_counter() - benchmark_start >= duration:
                    break
                log_duration_update()
                if len(active) == concurrency:
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
                active.add(
                    asyncio.create_task(
                        self._run_workload(
                            item,
                            benchmark_start=benchmark_start,
                            result=result,
                            stream=stream,
                            refresh=refresh,
                        )
                    )
                )
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
        result.wall_time = time.perf_counter() - benchmark_start
        log_summary(result, num_gpus=num_gpus)
        return result

    def sync_run(
        self,
        workload: list[Workload] | str,
        concurrency: int,
        duration: float,
        duration_update_interval: float | None = None,
        num_gpus: int | None = None,
        stream: bool = False,
    ) -> BenchmarkResult:
        try:
            return asyncio.run(
                self.run(
                    workload,
                    concurrency,
                    duration,
                    duration_update_interval=duration_update_interval,
                    num_gpus=num_gpus,
                    stream=stream,
                )
            )
        except KeyboardInterrupt:
            logger.warning("benchmark interrupted")
            assert self._result is not None and self._benchmark_start is not None
            result = self._result
            result.wall_time = time.perf_counter() - self._benchmark_start
            log_summary(result, num_gpus=num_gpus)
            return result
        finally:
            self._result = None
            self._benchmark_start = None
