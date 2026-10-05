#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --job-name=lucid-wc-transport-review
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/transport/cpu-%j.log
#SBATCH --chdir=/sdf/group/neutrino/youngsam/sim/LUCiD

set -euo pipefail
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/sdf/group/neutrino/youngsam/sim/LUCiD
export UV_CACHE_DIR=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/uv_cache
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --no-sync \
  --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python \
  python audit/wc_data_review_20261004/transport/reproduce_transport.py
