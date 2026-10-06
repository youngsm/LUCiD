#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=wc_jax0438
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/runtime_jax0438/slurm-%j.out
set -euo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
[[ "$SLURM_JOB_PARTITION" == milano || "$SLURM_JOB_PARTITION" == roma ]]
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/audit_wc_20261006/runtime_jax0438/uv_cache"
runner=/sdf/home/y/youngsam/.local/bin/uv
interpreter=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python
audit_runtime="$PWD/audit_wc_20261006/runtime_jax0438"
if [[ ! -x "$audit_runtime/venv/bin/python" ]]; then
    "$runner" venv --no-python-downloads --python "$interpreter" "$audit_runtime/venv"
fi
timeout 300 "$runner" pip install --python "$audit_runtime/venv/bin/python" \
    'jax==0.4.38' 'jaxlib==0.4.38' 'flax==0.10.2' 'optax==0.2.4' 'numpy==1.26.4' \
    scipy h5py uproot tqdm matplotlib seaborn plotly ipython ipywidgets scikit-learn
"$runner" pip freeze --python "$audit_runtime/venv/bin/python" > "$audit_runtime/requirements-resolved.txt"
"$audit_runtime/venv/bin/python" "$audit_runtime/run_checks.py"
