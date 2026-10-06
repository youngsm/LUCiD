#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --job-name=wc_geometry_repro
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/geometry/repro-%j.out
set -euo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export UV_CACHE_DIR="$PWD/audit_wc_20261006/uv_cache" OMP_NUM_THREADS=4
runner=/sdf/home/y/youngsam/.local/bin/uv
interpreter=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python
"$runner" run --offline --no-project --no-sync --python "$interpreter" python audit_wc_20261006/geometry/repro_reflection_selfhit.py
"$runner" run --offline --no-project --no-sync --python "$interpreter" python audit_wc_20261006/geometry/repro_reflection_selfhit.py --fixed
