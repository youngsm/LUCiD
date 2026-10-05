#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=wc-e2e-audit
#SBATCH --chdir=/sdf/group/neutrino/youngsam/sim/LUCiD
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/end_to_end/reference_%j.log
set -euo pipefail
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/sdf/group/neutrino/youngsam/sim/LUCiD
export UV_CACHE_DIR=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/uv_cache
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project \
    --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python \
    python audit/wc_data_review_20261004/end_to_end/reference_path.py
