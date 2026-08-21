#!/usr/bin/env bash
# Install surgeval from source and run the CPU-only reference examples.
#
# The harness core needs only a CPU (no GPU/CUDA/ROCm). These examples run the
# venv's own surgeval binary directly so nothing bleeds in from a project-wide
# environment. The closed-loop Lumen example runs only if seldinger-lumen is
# importable in that same venv; otherwise it is skipped.
#
# Usage:
#   ./scripts/quickstart.sh
#   VENV_DIR=/tmp/sv-venv PYTHON=3.11 OUT_ROOT=/tmp/run ./scripts/quickstart.sh
set -euo pipefail

PY="${PYTHON:-3.13}"
VENV_DIR="${VENV_DIR:-.venv-quickstart}"
OUT_ROOT="${OUT_ROOT:-/tmp/vector-quickstart}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null 2>&1 || {
    echo "uv is required: https://docs.astral.sh/uv/" >&2
    exit 1
}

echo "== creating venv ($VENV_DIR, python $PY) =="
uv venv --python "$PY" "$VENV_DIR"
uv pip install --python "$VENV_DIR/bin/python" -e ".[dev]"

PY_BIN="$VENV_DIR/bin/python"
SV_BIN="$VENV_DIR/bin/surgeval"
VIDEO_OUT="$OUT_ROOT/video"
COUNTER_OUT="$OUT_ROOT/counterfactual"
CVS_OUT="$OUT_ROOT/cvs"
LUMEN_OUT="$OUT_ROOT/lumen"
mkdir -p "$VIDEO_OUT" "$COUNTER_OUT" "$CVS_OUT" "$LUMEN_OUT"

echo
echo "== video single-turn prediction (with abstention support) =="
"$SV_BIN" run -t docs/examples/tasks/video-nextstep \
    -a docs/examples/agents/example-video-predictor \
    --out "$VIDEO_OUT"

echo
echo "== counterfactual consequence ranking + replay =="
"$SV_BIN" bind docs/examples/tasks/counterfactual-recovery \
    docs/examples/agents/example-counterfactual-world-model
"$SV_BIN" run -t docs/examples/tasks/counterfactual-recovery \
    -a docs/examples/agents/example-counterfactual-world-model \
    --out "$COUNTER_OUT"
"$SV_BIN" replay "$COUNTER_OUT"

echo
echo "== laparoscopic CVS detection through the video modality adapter =="
"$SV_BIN" bind docs/examples/tasks/laparoscopic-cholec-cvs \
    docs/examples/agents/example-cvs-detector
"$SV_BIN" run -t docs/examples/tasks/laparoscopic-cholec-cvs \
    -a docs/examples/agents/example-cvs-detector \
    --out "$CVS_OUT"
"$SV_BIN" replay "$CVS_OUT"

if "$PY_BIN" -c "import lumen" 2>/dev/null; then
    echo
    echo "== closed-loop Lumen policy (seldinger-lumen detected) =="
    "$SV_BIN" bind docs/examples/tasks/lumen-nav-safe \
        docs/examples/agents/seldingermed-lumen-linear
    "$SV_BIN" run -t docs/examples/tasks/lumen-nav-safe \
        -a docs/examples/agents/seldingermed-lumen-linear -n 3 \
        --out "$LUMEN_OUT"
else
    echo
    echo "== skipping closed-loop Lumen example: seldinger-lumen not installed =="
fi

echo
echo "done. artifacts under $OUT_ROOT"
