#!/usr/bin/env python3
"""Generate a single-turn 8k input / 1k output Trie workload."""

import argparse
import json
from pathlib import Path


DEFAULT_INPUT_PROMPT_LENGTH = 8192
DEFAULT_FINAL_ASSISTANT_RESPONSE_LENGTH = 1024
TRIE_ROOT = Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a JSONL Trie workload with one completion request per trace.",
    )
    parser.add_argument(
        "--num-traces",
        type=positive_int,
        required=True,
        help="Number of trace rows to write.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=TRIE_ROOT / "workloads/single_turn_8k_1k.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--input-prompt-length",
        type=positive_int,
        default=DEFAULT_INPUT_PROMPT_LENGTH,
        help="Initial prompt length in tokens.",
    )
    parser.add_argument(
        "--final-assistant-response-length",
        type=positive_int,
        default=DEFAULT_FINAL_ASSISTANT_RESPONSE_LENGTH,
        help="Final assistant response length in tokens.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = {
        "input_prompt_length": args.input_prompt_length,
        "assistant_response_length": [],
        "num_turns": 0,
        "tool_call_latency": [],
        "tool_call_output_length": [],
        "final_assistant_response_length": args.final_assistant_response_length,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for _ in range(args.num_traces):
            f.write(json.dumps(row, separators=(",", ":")))
            f.write("\n")

    print(
        f"Wrote {args.num_traces} traces to {args.out} "
        f"({args.input_prompt_length} input tokens, "
        f"{args.final_assistant_response_length} output tokens)."
    )


if __name__ == "__main__":
    main()
