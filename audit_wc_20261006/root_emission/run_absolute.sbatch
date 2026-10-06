#!/bin/bash
#SBATCH --job-name=wc-audit-absolute
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/root_emission/absolute-%j.log
set -euo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export UV_CACHE_DIR="$PWD/audit_wc_20261006/uv_cache"
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --no-sync \
  --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python \
  python audit_wc_20261006/root_emission/check_absolute_direct_light.py
