#!/usr/bin/env python3
"""Generate a synthetic Trie workload with configurable turn and token shapes."""

import json
from pathlib import Path

import chz

TRIE_ROOT = Path(__file__).resolve().parents[1]


@chz.chz
class GenerateArgs:
    num_traces: int = chz.field(doc="Number of trace rows to write.")
    out: str = chz.field(
        default=str(TRIE_ROOT / "workloads/generated_workload.jsonl"),
        doc="Output JSONL path.",
    )
    num_turns: int = chz.field(
        default=0,
        doc="Number of tool-use turns per trace (0 means single-turn).",
    )
    input_prompt_length: int = chz.field(
        default=8192,
        doc="Initial prompt length in tokens.",
    )
    assistant_response_length: tuple[int, ...] = chz.field(
        default=(1024,),
        doc=(
            "Per-turn assistant response lengths in tokens, comma separated. "
            "A single value applies to every turn."
        ),
    )
    tool_call_output_length: tuple[int, ...] = chz.field(
        default=(1024,),
        doc=(
            "Per-turn tool call output lengths in tokens, comma separated. "
            "A single value applies to every turn."
        ),
    )
    tool_call_latency: tuple[float, ...] = chz.field(
        default=(0.0,),
        doc=(
            "Per-turn simulated tool call latencies in seconds, comma separated. "
            "A single value applies to every turn."
        ),
    )
    final_assistant_response_length: int = chz.field(
        default=1024,
        doc="Final assistant response length in tokens.",
    )
    shared_prefix_length: int = chz.field(
        default=0,
        doc=(
            "Leading tokens of the input prompt shared across traces "
            "(e.g. system prompt + tool definitions). Part of input_prompt_length."
        ),
    )
    num_shared_prefixes: int = chz.field(
        default=1,
        doc="Number of distinct shared prefixes; traces are assigned round-robin.",
    )

    @chz.validate
    def _validate_fields(self) -> None:
        if self.num_traces <= 0:
            raise ValueError("num_traces must be greater than 0")
        if self.num_turns < 0:
            raise ValueError("num_turns must be greater than or equal to 0")
        if self.input_prompt_length <= 0:
            raise ValueError("input_prompt_length must be greater than 0")
        if not 0 <= self.shared_prefix_length <= self.input_prompt_length:
            raise ValueError(
                "shared_prefix_length must be between 0 and input_prompt_length"
            )
        if self.num_shared_prefixes <= 0:
            raise ValueError("num_shared_prefixes must be greater than 0")
        if self.final_assistant_response_length <= 0:
            raise ValueError("final_assistant_response_length must be greater than 0")
        if any(v <= 0 for v in self.assistant_response_length):
            raise ValueError("assistant_response_length values must be greater than 0")
        if any(v < 0 for v in self.tool_call_output_length):
            raise ValueError(
                "tool_call_output_length values must be greater than or equal to 0"
            )
        if any(v < 0 for v in self.tool_call_latency):
            raise ValueError(
                "tool_call_latency values must be greater than or equal to 0"
            )
        if self.num_turns > 0:
            for name in (
                "assistant_response_length",
                "tool_call_output_length",
                "tool_call_latency",
            ):
                count = len(getattr(self, name))
                if count not in (1, self.num_turns):
                    raise ValueError(
                        f"{name} expects 1 or {self.num_turns} values, got {count}"
                    )


def _per_turn(values: tuple, num_turns: int) -> list:
    if num_turns == 0:
        return []
    if len(values) == 1:
        return list(values) * num_turns
    return list(values)


def main() -> None:
    args = chz.entrypoint(GenerateArgs)
    row = {
        "input_prompt_length": args.input_prompt_length,
        "assistant_response_length": _per_turn(
            args.assistant_response_length, args.num_turns
        ),
        "num_turns": args.num_turns,
        "tool_call_latency": _per_turn(args.tool_call_latency, args.num_turns),
        "tool_call_output_length": _per_turn(
            args.tool_call_output_length, args.num_turns
        ),
        "final_assistant_response_length": args.final_assistant_response_length,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i in range(args.num_traces):
            if args.shared_prefix_length:
                row["shared_prefix_length"] = args.shared_prefix_length
                row["shared_prefix_id"] = f"prefix-{i % args.num_shared_prefixes}"
            f.write(json.dumps(row, separators=(",", ":")))
            f.write("\n")

    print(
        f"Wrote {args.num_traces} traces to {out} "
        f"({args.num_turns} turns, {args.input_prompt_length} input tokens, "
        f"{args.final_assistant_response_length} final output tokens)."
    )


if __name__ == "__main__":
    main()
