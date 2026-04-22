from dataclasses import dataclass
from typing import Any


@dataclass
class Turn:
    assistant_response_length: int
    tool_call_output_length: int
    tool_call_latency: float

    def __post_init__(self) -> None:
        if self.assistant_response_length < 0:
            raise ValueError("assistant_response_length must be non-negative")
        if self.tool_call_output_length < 0:
            raise ValueError("tool_call_output_length must be non-negative")
        if self.tool_call_latency < 0:
            raise ValueError("tool_call_latency must be non-negative")


@dataclass
class Workload:
    input_prompt_length: int
    turns: list[Turn]
    final_assistant_response_length: int

    def cumulative_prompt_tokens(self, through_turn: int) -> int:
        return self.input_prompt_length + sum(
            self.turns[i].assistant_response_length
            + self.turns[i].tool_call_output_length
            for i in range(through_turn)
        )

    @classmethod
    def from_jsonl_row(cls, d: dict[str, Any]) -> "Workload":
        num_turns = d["num_turns"]
        resp = d["assistant_response_length"]
        out = d["tool_call_output_length"]
        lat = d["tool_call_latency"]
        for name, values in (
            ("assistant_response_length", resp),
            ("tool_call_output_length", out),
            ("tool_call_latency", lat),
        ):
            if not isinstance(values, list):
                raise ValueError(f"{name} must be a list of length {num_turns}")
            if len(values) != num_turns:
                raise ValueError(
                    f"{name} length {len(values)} != num_turns {num_turns}"
                )
        turns = [Turn(resp[i], out[i], lat[i]) for i in range(num_turns)]
        return cls(
            input_prompt_length=d["input_prompt_length"],
            turns=turns,
            final_assistant_response_length=d["final_assistant_response_length"],
        )
