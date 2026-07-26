#!/usr/bin/env python3
"""Build a trie workload JSONL from captured request/response traffic.

1. Export a sample of traffic to JSONL, one object per line:
     {"input": "<chat messages JSON array>", "output": "<completion text>"}

2. python make_workload.py sample.jsonl workload.jsonl <hf-tokenizer-or-local-dir>

Each output trace models a session: a base prefix prefilled cold on turn 1,
then per-turn growth of (assistant response + new user/tool input) — which is
what produces realistic prefix-cache hit rates.

Convert target rpm to trie arrival_rate with:
  arrival_rate = rpm / 60 / avg_requests_per_trace   (printed at the end)
"""
import json
import sys

import numpy as np
from transformers import AutoTokenizer

sample_path, out_path, tok_name = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(tok_name)

rows = []
for line in open(sample_path):
    r = json.loads(line)
    try:
        msgs = json.loads(r["input"])
    except Exception:
        continue
    if not isinstance(msgs, list) or not msgs:
        continue
    try:
        tin = len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
    except Exception:
        tin = sum(len(tok.encode(str(m.get("content", "")))) for m in msgs)
    tlast = max(1, len(tok.encode(str(msgs[-1].get("content", "")))))
    tout = max(16, len(tok.encode(r.get("output", ""))))
    rows.append((tin, tlast, tout, len(msgs)))

traces, req_per_trace = [], []
for tin, tlast, tout, nmsg in rows:
    n_turns = int(np.clip(nmsg // 2, 1, 25))
    base = max(2000, tin - n_turns * (tout + tlast))
    traces.append({
        "input_prompt_length": base,
        "num_turns": n_turns,
        "assistant_response_length": [tout] * n_turns,
        "tool_call_output_length": [tlast] * n_turns,
        "tool_call_latency": [0.0] * n_turns,
        "final_assistant_response_length": tout,
    })
    req_per_trace.append(n_turns + 1)

with open(out_path, "w") as f:
    for t in traces:
        f.write(json.dumps(t) + "\n")

rpt = float(np.mean(req_per_trace))
tins = [t for t, _, _, _ in rows]
print(f"traces={len(traces)} avg_requests_per_trace={rpt:.1f} "
      f"input_tok p50={np.percentile(tins, 50):.0f} p95={np.percentile(tins, 95):.0f}")
for rpm in (25, 30, 50, 75, 125):
    print(f"  {rpm} rpm -> arrival_rate={rpm / 60 / rpt:.4f}")
