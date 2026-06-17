import logging

import chz
import structlog

from trie.client import Client


@chz.chz
class RunArgs:
    workload_path: str = chz.field(doc="Path to the JSONL workload file.")
    endpoint: str = chz.field(doc="OpenAI-compatible endpoint base URL.")
    model: str = chz.field(doc="Model name to send to the completion API.")
    concurrency: int | None = chz.field(
        default=None,
        doc="Maximum number of concurrent traces. Mutually exclusive with arrival_rate.",
    )
    arrival_rate: float | None = chz.field(
        default=None,
        doc="New trace arrivals per second. Mutually exclusive with concurrency.",
    )
    tokenizer_model: str | None = chz.field(
        default=None,
        doc="Optional model name override for loading tokenizer.",
    )
    duration: float = chz.field(
        default=300.0,
        doc="Duration in seconds to run the benchmark.",
    )
    duration_update_interval: float | None = chz.field(
        default=None,
        doc="Optional interval in seconds for logging remaining benchmark duration.",
    )
    seed: int | None = chz.field(
        default=None,
        doc="Optional random seed for reproducible workload resolution and prompt generation.",
    )
    api_key: str | None = chz.field(
        default=None,
        doc="API key for the endpoint. Falls back to OPENAI_API_KEY env var if not set.",
    )
    max_retries: int = chz.field(
        default=2,
        doc="Number of retries for the endpoint, openai default is 2.",
    )
    timeout: float = chz.field(
        default=600.0,
        doc="Timeout for the endpoint, openai default is 600 seconds.",
    )
    num_gpus: int | None = chz.field(
        default=None,
        doc="Optional GPU count used for the steady-state per-GPU throughput column.",
    )
    stream: bool = chz.field(
        default=False,
        doc="Use streaming completions and report TTFT, TTFAT, and TPS per trace.",
    )

    @chz.validate
    def _validate_fields(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be greater than 0")
        if (
            self.duration_update_interval is not None
            and self.duration_update_interval <= 0
        ):
            raise ValueError("duration_update_interval must be greater than 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if self.num_gpus is not None and self.num_gpus < 1:
            raise ValueError("num_gpus must be at least 1")


def main() -> None:
    args: RunArgs = chz.entrypoint(RunArgs)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    Client(
        endpoint=args.endpoint,
        model=args.model,
        tokenizer_model=args.tokenizer_model,
        api_key=args.api_key,
        seed=args.seed,
        max_retries=args.max_retries,
        timeout=args.timeout,
    ).sync_run(
        args.workload_path,
        concurrency=args.concurrency,
        arrival_rate=args.arrival_rate,
        duration=args.duration,
        duration_update_interval=args.duration_update_interval,
        num_gpus=args.num_gpus,
        stream=args.stream,
    )


if __name__ == "__main__":
    main()
