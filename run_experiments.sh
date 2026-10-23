#!/usr/bin/env bash
# =============================================================================
# AGA experiment runner (with the augmented-graph cache).
#
# Runs `python -m aga.main --train-compare` for one dataset across one or
# more GAD backbones. The FIRST model pays the LLM planning cost (Gap Analyzer
# + Strategy Planner) and writes the augmented graph to the cache; every
# subsequent model on the SAME config loads the cached graph and skips the
# LLM entirely (see aga/orchestration/cache.py).
#
# Usage:
#   ./run_experiments.sh [options] DATASET [MODEL...]
#
#   DATASET   Cora | Pubmed | Arxiv | History
#   MODEL     dominant cola gcn gat bwgnn bwgnn_hetero amnet h2fdetector
#             gdn care_gnn ...  or the special word `all`.
#
# Options:
#   --runs N         repeated train_compare runs, mean±std (default 5)
#   --backend NAME   LLM backend override (default: from the config, i.e.
#                    openai/Qwen3; use `--backend dummy` for keyless smoke runs)
#   --cache-dir DIR  augmented-graph cache dir (default cache/)
#   --no-cache       disable the cache for this invocation
#   --force-refresh  ignore the cached augmented graph, regenerate + overwrite
#   --config PATH    config file (default configs/default.yaml)
#   --data-dir DIR   dataset directory (default data/)
#   --report-only    plan + print only; no augmentation, no training
#   -h | --help      this help
#
# Examples:
#   ./run_experiments.sh Pubmed gcn                          # 1st run: generate + cache
#   ./run_experiments.sh Pubmed gcn bwgnn h2fdetector        # 2nd..4th: cache HIT, no LLM
#   ./run_experiments.sh History all                         # all backbones, one dataset
#   ./run_experiments.sh --backend dummy --runs 2 Cora all   # keyless quick check
#   ./run_experiments.sh --force-refresh Pubmed gcn          # re-sample the LLM plan
#
# Full per-model outputs are written to logs/<DATASET>_<MODEL>_<ts>.log;
# a summary table is printed at the end.
# =============================================================================
set -euo pipefail

# Always run from the repo root so data/ configs/ resolve, wherever invoked.
cd "$(dirname "$0")"

# --- python resolution -------------------------------------------------------
PY="${AGA_PYTHON:-/opt/miniconda3/envs/python312/bin/python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY="$(command -v python3 || true)"
fi
if [ -z "${PY:-}" ]; then
  echo "ERROR: no python found (set AGA_PYTHON=/path/to/python)." >&2
  exit 1
fi

# --- defaults ----------------------------------------------------------------
RUNS=5
BACKEND_ARGS=()
CACHE_DIR="cache/"
NO_CACHE=0
FORCE_REFRESH=0
CONFIG="configs/default.yaml"
DATA_DIR="data/"
REPORT_ONLY=0
ALL_MODELS=(dominant cola mlp graphsage gcn gat bwgnn bwgnn_hetero amnet h2fdetector gdn care_gnn pcgnn ghrn spacegnn gaga pmp)

usage() {
  cat <<'EOF'
Usage: ./run_experiments.sh [options] DATASET [MODEL...]

  DATASET   Cora | Pubmed | Arxiv | History
  MODEL     dominant cola gcn gat bwgnn bwgnn_hetero amnet h2fdetector
            gdn care_gnn ...  or the special word `all`.

Options:
  --runs N          repeated train_compare runs, mean±std (default 5)
  --backend NAME    LLM backend override (default: from the config, i.e.
                    openai/Qwen3; use `--backend dummy` for keyless smoke runs)
  --cache-dir DIR   augmented-graph cache dir (default cache/)
  --no-cache        disable the cache for this invocation
  --force-refresh   ignore the cached augmented graph, regenerate + overwrite
  --config PATH     config file (default configs/default.yaml)
  --data-dir DIR    dataset directory (default data/)
  --report-only     plan + print only; no augmentation, no training
  -h | --help       this help

Examples:
  ./run_experiments.sh Pubmed gcn                        # 1st run: generate + cache
  ./run_experiments.sh Pubmed gcn bwgnn h2fdetector      # 2nd..4th: cache HIT, no LLM
  ./run_experiments.sh History all                       # all backbones, one dataset
  ./run_experiments.sh --backend dummy --runs 2 Cora all # keyless quick check
  ./run_experiments.sh --force-refresh Pubmed gcn        # re-sample the LLM plan

The first model pays the LLM planning cost and writes the augmented-graph
cache; every subsequent model on the SAME config reuses it (no LLM calls).
Full per-model logs go to logs/<DATASET>_<MODEL>_<timestamp>.log; a summary
table is printed at the end.
EOF
}

# --- option parsing ----------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --runs)        RUNS="$2"; shift 2 ;;
    --backend)     BACKEND_ARGS=(--backend "$2"); shift 2 ;;
    --cache-dir)   CACHE_DIR="$2"; shift 2 ;;
    --no-cache)    NO_CACHE=1; shift ;;
    --force-refresh) FORCE_REFRESH=1; shift ;;
    --config)      CONFIG="$2"; shift 2 ;;
    --data-dir)    DATA_DIR="$2"; shift 2 ;;
    --report-only) REPORT_ONLY=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    --)            shift; break ;;
    -*)            echo "ERROR: unknown option $1 (see --help)" >&2; exit 1 ;;
    *)             break ;;   # first positional: DATASET
  esac
done

DATASET="${1:-}"
if [ -z "$DATASET" ]; then
  usage
  echo "ERROR: DATASET is required (Cora | Pubmed | Arxiv | History)." >&2
  exit 1
fi
shift || true

MODELS=("$@")
if [ "${#MODELS[@]}" -eq 0 ] || [ "${MODELS[0]-}" = "all" ]; then
  MODELS=("${ALL_MODELS[@]}")
fi

# --- assembled CLI args ------------------------------------------------------
CACHE_ARGS=(--cache-dir "$CACHE_DIR")
if [ "$NO_CACHE" = "1" ]; then
  CACHE_ARGS=(--cache-dir "")   # explicit empty string disables the cache
fi
REFRESH_ARGS=()
if [ "$FORCE_REFRESH" = "1" ]; then
  REFRESH_ARGS=(--force-refresh)
fi
TRAIN_ARGS=()
if [ "$REPORT_ONLY" = "1" ]; then
  TRAIN_ARGS=(--report-only)
else
  TRAIN_ARGS=(--train-compare)
fi

# --- run ---------------------------------------------------------------------
mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
BACKEND_DESC="config(→openai)"
if [ "${#BACKEND_ARGS[@]}" -gt 0 ]; then BACKEND_DESC="${BACKEND_ARGS[1]}"; fi

echo "======================================================================"
echo " AGA experiments: dataset=$DATASET models=${MODELS[*]}"
echo " runs=$RUNS backend=$BACKEND_DESC cache=$([ "$NO_CACHE" = 1 ] && echo OFF || echo "$CACHE_DIR") python=$PY"
echo "======================================================================"

FAILED=0
SUMMARY=()

for MODEL in "${MODELS[@]}"; do
  LOG="logs/${DATASET}_${MODEL}_${TS}.log"
  echo ""
  echo "[run] $DATASET / $MODEL  (log: $LOG)"
  if "$PY" -m aga.main \
      --config "$CONFIG" --data-dir "$DATA_DIR" \
      --dataset "$DATASET" --model "$MODEL" --runs "$RUNS" \
      ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} \
      ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} \
      ${REFRESH_ARGS[@]+"${REFRESH_ARGS[@]}"} \
      ${BACKEND_ARGS[@]+"${BACKEND_ARGS[@]}"} \
      >"$LOG" 2>&1; then
    # Pull the headline lines from the log for the final summary. Patterns are
    # anchored on "AUC=" so the earlier "augmented nodes : N" line can't shadow
    # the TRAINING COMPARISON rows; [[:space:]]/sed -E keep BSD (macOS) happy.
    CACHE_LINE="$(grep -m1 'augmented-graph cache' "$LOG" || echo 'augmented-graph cache : n/a')"
    BASE_LINE="$(grep -m1 -E '[[:space:]]baseline[[:space:]]+AUC=' "$LOG" || true)"
    AUG_LINE="$(grep -m1 -E '[[:space:]]augmented[[:space:]]+AUC=' "$LOG" || true)"
    BASE_AUC="$(printf '%s\n' "$BASE_LINE" | awk '{sub(/^AUC=/, "", $2); print $2}')"
    AUG_AUC="$(printf '%s\n' "$AUG_LINE" | awk '{sub(/^AUC=/, "", $2); print $2}')"
    CACHE_TAG="$(printf '%s\n' "$CACHE_LINE" | sed -nE 's/.*: (HIT|MISS).*/\1/p')"
    if [ "$REPORT_ONLY" = "1" ]; then
      SUMMARY+=("$(printf '%-12s %-4s %s' "$MODEL" "${CACHE_TAG:----}" "plan generated (report-only)")")
    else
      SUMMARY+=("$(printf '%-12s %-4s baseline %-24s augmented %-24s' \
        "$MODEL" "${CACHE_TAG:----}" "${BASE_AUC:-n/a}" "${AUG_AUC:-n/a}")")
    fi
  else
    echo "[run] $MODEL FAILED (see $LOG)"
    tail -5 "$LOG" || true
    SUMMARY+=("$(printf '%-12s FAILED (see %s)' "$MODEL" "$LOG")")
    FAILED=$((FAILED+1))
  fi
done

# --- summary -----------------------------------------------------------------
echo ""
echo "======================================================================"
echo " SUMMARY — $DATASET ($RUNS run(s), backend=$BACKEND_DESC)"
echo "======================================================================"
if [ "$REPORT_ONLY" = "1" ]; then
  printf '%-12s %s\n' "model" "status"
else
  printf '%-12s %-4s %-24s %s\n' "model" "cache" "baseline AUC" "augmented AUC"
fi
for LINE in "${SUMMARY[@]}"; do
  echo "$LINE"
done
echo "----------------------------------------------------------------------"
echo "Full logs: logs/${DATASET}_*_${TS}.log   (augmented-graph cache: $CACHE_DIR)"

[ "$FAILED" -gt 0 ] && { echo "$FAILED run(s) failed."; exit 1; }
exit 0
