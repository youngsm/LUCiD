#!/bin/bash
#SBATCH --job-name=wc-g4-emission-check
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/emission/followup-%j.log
set -euo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/emission
export PYTHONPATH=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export UV_CACHE_DIR="$PWD/uv_cache"
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project \
 --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python \
 python check_baseline_and_timestamps.py
