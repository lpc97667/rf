#!/usr/bin/env bash

# Run all four ablations on the GPU visible to this shell then print the
# analysis of whatever result files exist so far.

# ./run_ablations.sh
# ./run_ablations.sh --quick

# Then run python analyze.py when results for all GPUs to be compared are
# present for the cross-architecture comparison.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results logs

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" == "base" ]]; then
  source "$("${CONDA_EXE:-conda}" info --base)/etc/profile.d/conda.sh"
  conda activate "${RF_ENV:-rf}"
fi

ARCH=$(python -c "import ablation_common as c; print(c.arch_key())")
STAMP=$(date +%m-%d-%H-%M)
echo "running the R1--R4 ablations on '${ARCH}', logging to logs/${STAMP}-${ARCH}-*.log"

echo "=== parity check ==="
python check_parity.py 2>&1 | tee "logs/${STAMP}-${ARCH}-parity.log"

for exp in a1_precision a2_autotune a3_splitk a4_batch_invariance; do
  log="logs/${STAMP}-${ARCH}-${exp}.log"
  echo "=== ${exp} ==="
  python "${exp}.py" "$@" 2>&1 | tee "${log}"
done

echo
echo "=== analysis over results/ ==="
python analyze.py
