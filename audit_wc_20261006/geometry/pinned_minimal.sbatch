#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --job-name=wc_geometry_pinned
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/geometry/pinned-%j.out
set -euo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export UV_CACHE_DIR="$PWD/audit_wc_20261006/runtime_jax0438/uv_cache" OMP_NUM_THREADS=4
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --no-sync --python "$PWD/audit_wc_20261006/runtime_jax0438/venv/bin/python" python audit_wc_20261006/geometry/find_minimal_pinned.py
