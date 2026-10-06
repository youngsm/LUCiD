#!/bin/bash
#SBATCH --job-name=wc-physics-check
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/physics_crosscheck/cpu-%j.log
set -euo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD"
export UV_CACHE_DIR=/tmp/youngsam-wc-physics-uv-cache
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --no-sync --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python audit_wc_20261006/physics_crosscheck/verify_optical_reference.py
