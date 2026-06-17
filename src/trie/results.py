from openai.types.completion_usage import CompletionUsage

from bisect import bisect_left
from dataclasses import dataclass, field

from trie.types import Workload


@dataclass
class CompletionPoint:
    """Cumulative token counts at a point in time, derived from the synthetic
    workload accounting (not from server usage fields; see ServerMetrics for server-reported counters)."""

    timestamp: float
    cumulative_completion_tokens: int = 0
    cumulative_new_prompt_tokens: int = 0
    cumulative_cached_prompt_tokens: int = 0

    @property
    def cumulative_prompt_tokens(self) -> int:
        return self.cumulative_new_prompt_tokens + self.cumulative_cached_prompt_tokens


@dataclass
class ThroughputRates:
    """Token throughput rates for each token category."""

    prompt_tok_s: float
    cached_tok_s: float
    new_prompt_tok_s: float
    completion_tok_s: float


def _find_points_from(
    points: list[CompletionPoint], start_timestamp: float
) -> tuple[CompletionPoint, CompletionPoint] | None:
    if len(points) < 2:
        return None
    idx = bisect_left(points, start_timestamp, key=lambda p: p.timestamp)
    if idx >= len(points) - 1:
        return None
    start, end = points[idx], points[-1]
    if start.timestamp == end.timestamp:
        return None
    return start, end


def _throughput_rates(
    points: tuple[CompletionPoint, CompletionPoint] | None,
) -> ThroughputRates:
    if points is None:
        return ThroughputRates(0.0, 0.0, 0.0, 0.0)
    start, end = points
    dt = end.timestamp - start.timestamp
    return ThroughputRates(
        prompt_tok_s=(end.cumulative_prompt_tokens - start.cumulative_prompt_tokens)
        / dt,
        cached_tok_s=(
            end.cumulative_cached_prompt_tokens - start.cumulative_cached_prompt_tokens
        )
        / dt,
        new_prompt_tok_s=(
            end.cumulative_new_prompt_tokens - start.cumulative_new_prompt_tokens
        )
        / dt,
        completion_tok_s=(
            end.cumulative_completion_tokens - start.cumulative_completion_tokens
        )
        / dt,
    )


@dataclass
class ServerMetrics:
    """Server-reported metrics for a single trace."""

    prompt_tokens: int = 0
    cached_tokens: int = 0
    eligible_prompt_tokens: int = 0
    _next_request_eligible_prompt_tokens: int = field(default=0, init=False, repr=False)

    def record_usage(self, usage: CompletionUsage) -> None:
        self.prompt_tokens += usage.prompt_tokens
        self.cached_tokens += (
            usage.prompt_tokens_details.cached_tokens
            if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens
            else 0
        )
        self.eligible_prompt_tokens += self._next_request_eligible_prompt_tokens
        self._next_request_eligible_prompt_tokens = (
            usage.prompt_tokens + usage.completion_tokens
        )

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def eligible_cache_hit_rate(self) -> float:
        return (
            self.cached_tokens / self.eligible_prompt_tokens
            if self.eligible_prompt_tokens
            else 0.0
        )


@dataclass
class BenchmarkResult:
    """Aggregated metrics from a benchmark run."""

    wall_time: float = 0.0
    completed_requests: int = 0
    failed_requests: int = 0
    latencies: list[float] = field(default_factory=list)
    ttfts: list[float] = field(default_factory=list)
    ttfats: list[float] = field(default_factory=list)
    tps_values: list[float] = field(default_factory=list)
    inter_token_latencies_ms: list[float] = field(default_factory=list)
    points: list[CompletionPoint] = field(default_factory=list)
    server_metrics: list[ServerMetrics] = field(default_factory=list)

    def _append_point(
        self,
        timestamp: float,
        completion_tokens: int = 0,
        new_prompt_tokens: int = 0,
        cached_prompt_tokens: int = 0,
    ) -> None:
        prev = self.points[-1] if self.points else CompletionPoint(0.0)
        self.points.append(
            CompletionPoint(
                timestamp=timestamp,
                cumulative_completion_tokens=prev.cumulative_completion_tokens
                + completion_tokens,
                cumulative_new_prompt_tokens=prev.cumulative_new_prompt_tokens
                + new_prompt_tokens,
                cumulative_cached_prompt_tokens=prev.cumulative_cached_prompt_tokens
                + cached_prompt_tokens,
            )
        )

    def record_turn(
        self,
        workload: Workload,
        turn_index: int,
        timestamp: float,
    ) -> None:
        new_prompt_tokens = (
            workload.input_prompt_length
            if turn_index == 0
            else workload.turns[turn_index - 1].tool_call_output_length
        )
        self._append_point(
            timestamp=timestamp,
            completion_tokens=workload.turns[turn_index].assistant_response_length,
            new_prompt_tokens=new_prompt_tokens,
            cached_prompt_tokens=workload.cumulative_prompt_tokens(turn_index)
            - new_prompt_tokens,
        )

    def record_success(
        self,
        workload: Workload,
        latency: float,
        timestamp: float,
    ) -> None:
        new_prompt_tokens = (
            workload.turns[-1].tool_call_output_length
            if workload.turns
            else workload.input_prompt_length
        )
        self._append_point(
            timestamp=timestamp,
            completion_tokens=workload.final_assistant_response_length,
            new_prompt_tokens=new_prompt_tokens,
            cached_prompt_tokens=workload.cumulative_prompt_tokens(len(workload.turns))
            - new_prompt_tokens,
        )
        self.latencies.append(latency)
        self.completed_requests += 1

    def record_failure(self) -> None:
        self.failed_requests += 1

    @property
    def completed_prompt_tokens(self) -> int:
        return self.points[-1].cumulative_prompt_tokens if self.points else 0

    @property
    def completed_completion_tokens(self) -> int:
        return self.points[-1].cumulative_completion_tokens if self.points else 0

    @property
    def new_prompt_tokens(self) -> int:
        return self.points[-1].cumulative_new_prompt_tokens if self.points else 0

    @property
    def cached_prompt_tokens(self) -> int:
        return self.points[-1].cumulative_cached_prompt_tokens if self.points else 0

    def last_30s_rates(self) -> ThroughputRates:
        last_ts = self.points[-1].timestamp if self.points else 0.0
        return _throughput_rates(_find_points_from(self.points, last_ts - 30.0))

    def steady_state_rates(self, warmup_frac: float = 0.2) -> ThroughputRates:
        last_ts = self.points[-1].timestamp if self.points else 0.0
        return _throughput_rates(_find_points_from(self.points, last_ts * warmup_frac))


@dataclass
class StreamAccumulator:
    """Accumulates per-request stream metrics for a single agent invocation."""

    _first_ttft: float | None = field(default=None, init=False)
    _last_first_token_offset_s: float | None = field(default=None, init=False)
    _sum_turn_tps: float = field(default=0.0, init=False)
    _inter_token_latencies_ms: list[float] = field(default_factory=list, init=False)
    _request_count: int = field(default=0, init=False)

    def record_turn_stream_metrics(
        self,
        ttft: float,
        latency: float,
        completion_tokens: int,
        *,
        first_token_offset_s: float | None = None,
        inter_token_latencies_ms: list[float] | None = None,
    ) -> None:
        if self._first_ttft is None:
            self._first_ttft = ttft
        self._last_first_token_offset_s = (
            ttft if first_token_offset_s is None else first_token_offset_s
        )
        if inter_token_latencies_ms:
            self._inter_token_latencies_ms.extend(inter_token_latencies_ms)
        decode_time_s = max(latency - ttft, 0.0)
        self._sum_turn_tps += (
            (completion_tokens - 1) / decode_time_s
            if completion_tokens > 1 and decode_time_s > 0
            else 0.0
        )
        self._request_count += 1

    def commit(self, result: BenchmarkResult) -> None:
        if self._request_count == 0:
            return
        assert self._first_ttft is not None
        assert self._last_first_token_offset_s is not None
        result.ttfts.append(self._first_ttft)
        result.ttfats.append(self._last_first_token_offset_s)
        result.tps_values.append(self._sum_turn_tps / self._request_count)
        result.inter_token_latencies_ms.extend(self._inter_token_latencies_ms)
