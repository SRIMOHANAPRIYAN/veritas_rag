#!/usr/bin/env bash
# Watchdog wrapper for the Gate 3 eval. Kills + restarts the eval if it wedges
# on a hung llama.cpp decode (the run_gate3_eval.py resume logic makes restarts
# cheap). A question that stalls twice in a row is added to the skip list.
#
# Usage (from repo root, venv active):
#   bash scripts/run_gate3_watchdog.sh
set -u

JSONL="evaluation/benchmarks/results_phase3.jsonl"
CUR="evaluation/benchmarks/gate3_current.txt"
SKIP="evaluation/benchmarks/gate3_skip.txt"
STALL=360        # seconds with no new JSONL row = wedged
TARGET=200       # total questions (scored + skipped) to finish
touch "$SKIP"
last_stalled=""

while true; do
  total=$(wc -l < "$JSONL" 2>/dev/null || echo 0)
  if [ "$total" -ge "$TARGET" ]; then
    echo ">> All $TARGET questions accounted for. Finished."
    break
  fi

  echo ">> Starting eval ($(date '+%H:%M:%S')) — $total/$TARGET done so far..."
  PYTHONPATH=. python scripts/run_gate3_eval.py &
  PID=$!

  prev=$(wc -l < "$JSONL" 2>/dev/null || echo 0)
  prev_t=$(date +%s)
  while kill -0 "$PID" 2>/dev/null; do
    sleep 30
    cur=$(wc -l < "$JSONL" 2>/dev/null || echo 0)
    if [ "$cur" -gt "$prev" ]; then
      prev=$cur; prev_t=$(date +%s)
    elif [ $(( $(date +%s) - prev_t )) -ge "$STALL" ]; then
      stuck=$(cat "$CUR" 2>/dev/null || echo "")
      echo ">> STALL (${STALL}s no progress) on question: $stuck"
      if [ -n "$stuck" ] && [ "$stuck" = "$last_stalled" ]; then
        echo "$stuck" >> "$SKIP"
        echo ">> $stuck stalled twice — added to skip list."
      fi
      last_stalled="$stuck"
      kill -9 "$PID" 2>/dev/null
      wait "$PID" 2>/dev/null
      break
    fi
  done
  wait "$PID" 2>/dev/null
  sleep 2   # let Metal release before restart
done

echo ">> Done. Summary:"
cat evaluation/benchmarks/results_phase3.json
