#!/bin/bash
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/transport
export UV_CACHE_DIR="$PWD/uv_cache" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python analyze_em_channels.py
