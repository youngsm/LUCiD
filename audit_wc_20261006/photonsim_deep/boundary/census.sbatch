#!/bin/bash
#SBATCH --job-name=wc-boundary-census
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/boundary/census-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/audit_wc_20261006/photonsim_deep/boundary/uv_cache"
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --no-sync --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python audit_wc_20261006/photonsim_deep/boundary/census_histories.py
