#!/usr/bin/env bash
# Endpoint validation: smoke + steady-rate SLA check + optional ramp.
# Usage:
#   ./validate_endpoint.sh <base_url> <model> <api_key> <workload.jsonl> <tokenizer_dir_or_hf_id> [rpm] [ramp]
# Examples:
#   ./validate_endpoint.sh https://inference-usw2.appliedcompute.com/nvidia/GLM-5.2-NVFP4 \
#       nvidia/GLM-5.2-NVFP4 $KEY shopify_workload.jsonl ./glm-tokenizer 25
#   ... 25 ramp   # adds 2x/3x/5x rate steps after the steady phase
set -euo pipefail
BASE=$1; MODEL=$2; KEY=$3; WORKLOAD=$4; TOKENIZER=$5; RPM=${6:-25}; RAMP=${7:-}

echo "== 1/3 smoke: tiny request"
curl -sf --max-time 30 -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}" \
  -o /dev/null -w "   code=%{http_code} total=%{time_total}s\n"

echo "== 2/3 smoke: production-sized request (~150KB body)"
python3 - "$BASE" "$MODEL" "$KEY" <<'EOF'
import json, sys, time, urllib.request
base, model, key = sys.argv[1:4]
body = json.dumps({"model": model, "max_tokens": 50, "stream": False,
    "messages": [{"role": "user", "content": "ctx: " + ("lorem ipsum " * 12000) + " Summarize in one word."}]}).encode()
req = urllib.request.Request(base + "/v1/chat/completions", data=body,
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=90) as r:
    r.read()
    print(f"   code={r.status} total={time.time()-t0:.2f}s body={len(body)}B")
EOF

# rpm -> traces/s using avg requests/trace from the workload
ARRIVAL=$(python3 - "$WORKLOAD" "$RPM" <<'EOF'
import json, sys
rpt = [json.loads(l)["num_turns"] + 1 for l in open(sys.argv[1])]
print(f"{float(sys.argv[2]) / 60 / (sum(rpt)/len(rpt)):.5f}")
EOF
)
echo "== 3/3 trie steady phase: ${RPM} rpm (arrival_rate=${ARRIVAL}), 10 min"
run_trie() {
  trie workload_path="$WORKLOAD" endpoint="$BASE/v1" model="$MODEL" \
    tokenizer_model="$TOKENIZER" api_key="$KEY" stream=True \
    arrival_rate="$1" duration="$2" duration_update_interval=120
}
run_trie "$ARRIVAL" 600

if [ "$RAMP" = "ramp" ]; then
  for mult in 2 3 5; do
    rate=$(python3 -c "print(f'{$ARRIVAL * $mult:.5f}')")
    echo "== ramp step ${mult}x (arrival_rate=${rate}), 5 min"
    run_trie "$rate" 300
  done
fi
echo "== done. Judge TTFT/latency percentiles above against the SLA; any failed_requests>0 is a red flag."
