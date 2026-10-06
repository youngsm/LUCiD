#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=wc_charge_audit
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/digitizer/high_charge_%j.log
set -euo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD" UV_CACHE_DIR="$PWD/audit_wc_20261006/uv_cache"
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python audit_wc_20261006/digitizer/probe_high_charge_response.py
