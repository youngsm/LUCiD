#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --job-name=wc-audit-stats
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/simulator_statistics/slurm-%j.out
set -euo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu PYTHONPATH="$PWD" UV_CACHE_DIR=/tmp/wc-audit-stats-uv
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python audit_wc_20261006/simulator_statistics/check_statistics.py
