#!/usr/bin/env bash
# Smoke tests for vanilla-rag. Runs against a local `wrangler dev` on :8787
# Usage: BASE=http://localhost:8787 bash tests/smoke.sh
set -euo pipefail
BASE="${BASE:-http://localhost:8787}"
pass=0; fail=0
check(){ if [ "$1" = "$2" ]; then echo "PASS: $3"; pass=$((pass+1)); else echo "FAIL: $3 (got '$1' want '$2')"; fail=$((fail+1)); fi; }

# 1. Health
h=$(curl -s "$BASE/api/health" | python3 -c "import sys,json;print(json.load(sys.stdin).get('status'))")
check "$h" "healthy" "health endpoint"

# 2. Stats
s=$(curl -s "$BASE/api/stats" | python3 -c "import sys,json;print('ok' if 'documents' in json.load(sys.stdin) else 'bad')")
check "$s" "ok" "stats endpoint"

# 3. Ingest
code=$(curl -s -o /tmp/ing.json -w "%{http_code}" -X POST "$BASE/api/documents" -H 'Content-Type: application/json' -d '{"filename":"t.txt","text":"Retrieval augmented generation grounds answers in documents. Cosine similarity ranks chunks."}')
check "$code" "201" "ingest document"
n=$(python3 -c "import json;print(json.load(open('/tmp/ing.json'))['chunk_count'])")
if [ "$n" -ge 1 ]; then echo "PASS: chunk count >=1"; pass=$((pass+1)); else echo "FAIL: chunk count"; fail=$((fail+1)); fi

# 4. Search
r=$(curl -s -X POST "$BASE/api/search" -H 'Content-Type: application/json' -d '{"query":"what grounds answers?"}' | python3 -c "import sys,json;print(len(json.load(sys.stdin)['results']))")
if [ "$r" -ge 1 ]; then echo "PASS: search returns results"; pass=$((pass+1)); else echo "FAIL: search empty"; fail=$((fail+1)); fi

# 5. Malformed input
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/api/query" -H 'Content-Type: application/json' -d '{"query":""}')
check "$code" "400" "empty query rejected"

# 6. Query (needs GROQ_API_KEY; skips gracefully)
q=$(curl -s -X POST "$BASE/api/query" -H 'Content-Type: application/json' -d '{"query":"What does RAG do?"}')
if echo "$q" | grep -q '"answer"'; then echo "PASS: query returns answer"; pass=$((pass+1)); else echo "SKIP/FAIL: query"; fail=$((fail+1)); fi

echo "---- $pass passed, $fail failed ----"
[ "$fail" -eq 0 ]
