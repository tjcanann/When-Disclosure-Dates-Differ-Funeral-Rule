#!/usr/bin/env bash
# Reproduce every table and every number in
# "When Disclosure Dates and Prices Diverge: Evidence from the Funeral Rule" (T. J. Canann).
# Usage: bash run_all.sh            (Python steps; LaTeX build runs if latexmk is installed)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
ROOT="$PWD"
export AUSTIN_PIPELINE="$ROOT/austin_pipeline" KC_DIR="$ROOT/kansas_city"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
mkdir -p logs letter/output letter/manuscript/tables

echo "[1/3] Austin pipeline: audited panel, corrections, base outputs"
cd "$AUSTIN_PIPELINE"
for s in apply_source_corrections.py replicate_CEP_phase3d.py revision_analysis.py adversarial_analysis.py; do
  echo "  $s"; "$PY" "$s" > "$ROOT/logs/$s.log" 2>&1 || { echo "FAILED: $s (see logs/$s.log)"; exit 1; }
done

echo "[2/3] Letter analysis, simulation, appendix diagnostics, tables"
cd "$ROOT/letter"
"$PY" el_analysis.py "$AUSTIN_PIPELINE" > "$ROOT/logs/el_analysis.log" 2>&1
"$PY" sim_and_bootstrap.py              > "$ROOT/logs/sim_and_bootstrap.log" 2>&1
"$PY" el_appendix.py                    > "$ROOT/logs/el_appendix.log" 2>&1
"$PY" build_el_tables.py                > "$ROOT/logs/build_el_tables.log" 2>&1
"$PY" write_appendix_body.py            > "$ROOT/logs/write_appendix_body.log" 2>&1

echo "[3/3] Manuscript (if LaTeX is available) and number checks"
if command -v latexmk >/dev/null 2>&1; then
  (cd manuscript && for t in el ssrn appendix; do
     latexmk -pdf -interaction=nonstopmode "$t.tex" > "$ROOT/logs/latex_$t.log" 2>&1 || { echo "LaTeX failed: $t"; exit 1; }
   done)
  if command -v pdftotext >/dev/null 2>&1 && command -v detex >/dev/null 2>&1; then "$PY" wordcount.py > "$ROOT/logs/wordcount.log"; fi
else
  echo "  latexmk not found: skipping PDF build (tables are in letter/manuscript/tables)"
fi
"$PY" check_numbers.py > "$ROOT/logs/check_numbers.log" 2>&1 || { tail -5 "$ROOT/logs/check_numbers.log"; echo "NUMBER CHECKS FAILED"; exit 1; }
tail -1 "$ROOT/logs/check_numbers.log"
echo "Done."
